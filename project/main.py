import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.1)
        return out


# Squeeze-and-Excitation Block
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # kernel_size를 명시적으로 (H, W)로 설정
        kernel_size = (x.size(2), x.size(3))
        out = F.avg_pool2d(x, kernel_size).view(batch_size, channels)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out)).view(batch_size, channels, 1, 1)
        return x * out


# Camera Control Model
class CameraControlModel(nn.Module):
    def __init__(self):
        super(CameraControlModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SqueezeExcitationBlock(32)
        self.pool1 = nn.MaxPool2d(2)

        self.res2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.res3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.res4 = ResidualBlock(128, 256)
        self.se2 = SqueezeExcitationBlock(256)
        self.pool4 = nn.MaxPool2d(2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = self.se1(x)
        x = self.pool1(x)

        x = self.res2(x)
        x = self.pool2(x)

        x = self.res3(x)
        x = self.pool3(x)

        x = self.res4(x)
        x = self.se2(x)
        x = self.pool4(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = F.dropout(x, 0.3)
        x = torch.tanh(self.fc_out(x))
        return x


# Custom Dataset
class CameraControlDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


def load_labeled_data(csv_path, image_folder, empty_folder, img_size=(128, 128)):
    labeled_data = pd.read_csv(csv_path)

    images = []
    labels = []

    # 공이 있는 이미지와 좌표 불러오기
    for _, row in labeled_data.iterrows():
        image_file = row['image']
        x, y = row['x'], row['y']

        img_path = str(os.path.join(image_folder, image_file))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 컬러 이미지로 로드
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = img / 255.0  # [0, 255] 범위의 이미지를 [0, 1] 범위로 정규화
        images.append(img)

        # 좌표 정규화 (0~128 사이 값을 -1~1 사이로 변환)
        norm_x = (x / img_size[0]) * 2 - 1
        norm_y = (y / img_size[1]) * 2 - 1
        labels.append([norm_x, norm_y])

    # 공이 없는 이미지에 대해서는 (0, 0)을 라벨로 할당
    for empty_file in os.listdir(empty_folder):
        empty_img_path = str(os.path.join(empty_folder, empty_file))
        img = cv2.imread(empty_img_path, cv2.IMREAD_COLOR)  # 공이 없는 이미지도 컬러로 로드
        if img is None:
            print(f"Unable to load image: {empty_img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = img / 255.0  # [0, 255] 범위의 이미지를 [0, 1] 범위로 정규화
        images.append(img)
        labels.append([0, 0])  # 공이 없을 때는 (0, 0) 출력

    return np.array(images), np.array(labels)


# 이미지 저장 함수
def save_result_image(image, predicted, target, idx, save_dir='./results'):
    # 이미지를 numpy 형식으로 변환
    img = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)로 순서 변경 및 CPU로 이동

    # 이미지를 [0, 255]로 스케일링하고, np.uint8로 변환
    img = (img * 255).astype(np.uint8)

    # 만약 이미지가 grayscale로 잘못된 경우 (1채널), 이를 3채널로 확장
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 예측된 좌표와 실제 좌표를 (0~128) 크기로 변환
    pred_x = int((predicted[0].item() + 1) * 64)
    pred_y = int((predicted[1].item() + 1) * 64)
    target_x = int((target[0].item() + 1) * 64)
    target_y = int((target[1].item() + 1) * 64)

    # 예측 좌표 (빨간색) 및 실제 좌표 (초록색) 표시
    img = cv2.circle(img, (pred_x, pred_y), 5, (0, 0, 255), -1)
    img = cv2.circle(img, (target_x, target_y), 5, (0, 255, 0), -1)

    # 이미지 저장 경로
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"result_{idx}.png")
    cv2.imwrite(img_path, img)
    print(f"Result image saved: {img_path}")


def train_camera_control_model(model, train_loader, val_loader, device, epochs=50, model_save_path='./model', results_dir='./results'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    # 모든 Epoch이 끝날 때 검증 데이터에서 예측 결과 저장
    for idx, (images, labels) in enumerate(val_loader):
        image, label = images.to(device), labels.to(device)
        output = model(image)
        for i in range(image.size(0)):  # 배치 내 각 이미지에 대해 결과 저장
            save_result_image(image[i], output[i], label[i], idx * len(images) + i, save_dir=results_dir)

    # 모델 저장 경로 생성
    os.makedirs(model_save_path, exist_ok=True)

    # 학습 완료 후 모델을 ONNX 형식으로 저장
    onnx_path = os.path.join(model_save_path, 'camera_control_model.onnx')

    # 데이터셋에서 하나의 샘플을 사용해 ONNX로 변환
    sample_img, _ = next(iter(train_loader))  # 첫 번째 배치에서 샘플 가져옴
    sample_img = sample_img[0:1].to(device)  # 배치에서 첫 이미지만 사용

    torch.onnx.export(model, sample_img, onnx_path, opset_version=11)
    print(f"Model saved as ONNX format at: {onnx_path}")


if __name__ == "__main__":
    train_csv = './data/train_labels.csv'
    train_folder = './data/train/balls'
    empty_folder = './data/train/empty'
    test_csv = './data/test_labels.csv'
    test_folder = './data/test/balls'
    test_empty_folder = './data/test/empty'

    # 데이터셋 로드
    train_images, train_labels = load_labeled_data(train_csv, train_folder, empty_folder)
    test_images, test_labels = load_labeled_data(test_csv, test_folder, test_empty_folder)

    # 이미지 전처리 (PyTorch 텐서 변환)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 데이터셋 및 데이터 로더 생성
    train_dataset = CameraControlDataset(train_images, train_labels, transform=transform)
    test_dataset = CameraControlDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # MPS 디바이스 확인 및 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} for training.")

    # 모델 초기화
    model = CameraControlModel()

    # 모델 학습 및 ONNX로 저장
    train_camera_control_model(model, train_loader, val_loader, device, epochs=50, model_save_path='./model')
