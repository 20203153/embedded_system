import os
from typing import Tuple
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from numpy import ndarray
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from tqdm import tqdm

# Depthwise Separable Convolution Layer
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.depthwise(x))
        x = self.pointwise(x)
        x = self.bn2(x)
        return x

class CameraControlNet(nn.Module):
    def __init__(self):
        super(CameraControlNet, self).__init__()

        # Convolutional Layers
        self.conv1 = DepthwiseSeparableConv(3, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = DepthwiseSeparableConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DepthwiseSeparableConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = DepthwiseSeparableConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)

        # Output Layers
        self.fc_out_angles = nn.Linear(256, 2)  # Output angles (x, y)
        self.fc_out_ball_detect = nn.Linear(256, 1)  # Output for ball detection (0 or 1)

        # Activation functions
        self.relu = nn.Mish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x))) 
        x = self.pool2(self.relu(self.conv2(x))) 
        x = self.pool3(self.relu(self.conv3(x))) 
        x = self.pool4(self.relu(self.conv4(x))) 

        x = x.view(-1, 256 * 16 * 16)  # Flatten
        x = self.dropout1(self.relu(self.fc1(x)))  
        x = self.dropout2(self.relu(self.fc2(x)))  

        angles = self.fc_out_angles(x)  # Angles output
        ball_detect = self.sigmoid(self.fc_out_ball_detect(x))  # Ball detection output

        return angles, ball_detect

# 데이터셋 클래스 정의
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

def load_labeled_data(csv_path, image_folder, empty_folder, img_size=(256, 256)):
    labeled_data = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in labeled_data.iterrows():
        image_file = row['image']
        x, y = row['x'], row['y']

        img_path = str(os.path.join(image_folder, image_file))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  

        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        images.append(img)

        norm_x = (x / img_size[0]) * 2 - 1
        norm_y = (y / img_size[1]) * 2 - 1
        labels.append([norm_x, norm_y, 1])  # (x, y, has_ball)

    for empty_file in os.listdir(empty_folder):
        empty_img_path = str(os.path.join(empty_folder, empty_file))
        img = cv2.imread(empty_img_path, cv2.IMREAD_COLOR)  
        if img is None:
            print(f"Unable to load image: {empty_img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        labels.append([0, 0, 0])  # (0, 0, 0) for no ball

    return images, labels

# 이미지 저장 함수
def save_result_image(image, predicted, target, idx, save_dir='./results', size=256):
    img: ndarray = image.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pred_x = int((predicted[0].item() + 1) * (size // 2))
    pred_y = int((predicted[1].item() + 1) * (size // 2))
    target_x = int((target[0].item() + 1) * (size // 2))
    target_y = int((target[1].item() + 1) * (size // 2))

    img = cv2.circle(img, (pred_x, pred_y), 5, (0, 0, 255), -1)
    img = cv2.circle(img, (target_x, target_y), 5, (0, 255, 0), -1)

    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"result_{idx}.png")
    cv2.imwrite(img_path, img)
    print(f"Result image saved: {img_path}")

def train_camera_control_model(model, train_loader, val_loader, device, epochs=100, lr=1e-4, model_save_path='./model', results_dir='./results'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion_angle = nn.MSELoss()
    criterion_ball = nn.BCELoss()  # Binary cross entropy for ball detection

    os.makedirs(model_save_path, exist_ok=True)  # 모델 저장 경로 생성

    for epoch in tqdm(range(epochs), "Epoch", position=1):
        model.train()
        running_loss_angle = 0.0
        running_loss_ball = 0.0
        for images, labels in tqdm(train_loader, "Batch", position=0):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs_angles, outputs_ball = model(images)

            # Angles loss
            loss_angle = criterion_angle(outputs_angles, labels[:, :2])  # 첫 두 개의 라벨
            # Ball detection loss
            loss_ball = criterion_ball(outputs_ball, labels[:, 2:])  # 세 번째 라벨

            # Total loss
            loss = loss_angle + loss_ball
            loss.backward()
            optimizer.step()

            running_loss_angle += loss_angle.item()
            running_loss_ball += loss_ball.item()

        scheduler.step()

        model.eval()
        val_loss_angle = 0.0
        val_loss_ball = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs_angles, outputs_ball = model(images)

                val_loss_angle += criterion_angle(outputs_angles, labels[:, :2]).item()
                val_loss_ball += criterion_ball(outputs_ball, labels[:, 2:]).item()

        print(f"Epoch [{epoch + 1}/{epochs}], Angle Loss: {running_loss_angle / len(train_loader):.4f}, Ball Loss: {running_loss_ball / len(train_loader):.4f}")

    # 모델 저장
    torch.save(model.state_dict(), os.path.join(model_save_path, "fine_tuned_model.pth"))
    print(f"Model saved at {model_save_path}")

    # 검증 데이터에서 예측 결과 저장
    for idx, (images, labels) in tqdm(enumerate(val_loader), "Gen image", position=0):
        images, labels = images.to(device), labels.to(device)
        outputs_angles, outputs_ball = model(images)
        
        for i in tqdm(range(images.size(0)), "Batch", position=1):
            # 공 여부를 판단하기 위한 값
            ball_label = labels[i, 2].item()  # 실제 레이블에서 공 존재 여부
            ball_detected = (outputs_ball[i].item() > 0.5, ball_label)  # 예측 유무와 실제 레이블을 튜플로 저장
            
            # save_result_image 호출
            save_result_image(
                images[i],
                outputs_angles[i],
                labels[i, :2],  # 각도
                ball_detected,  # Tuple: (예측 좌표, 실제 유무)
                idx * len(images) + i,
                save_dir=results_dir
            )

    # ONNX 모델 저장
    onnx_path = os.path.join(model_save_path, 'camera_control_model.onnx')
    sample_img, _ = next(iter(train_loader))
    sample_img = sample_img[0:1].to(device)

    torch.onnx.export(model, sample_img, onnx_path, opset_version=11)
    print(f"Model saved as ONNX format at: {onnx_path}")

def save_result_image(image, predicted_angles, target_angles, ball_detected: Tuple[bool, bool], idx, save_dir='./results', size=256):
    """이미지를 저장하고 예측된 각도를 그려줍니다."""
    img: ndarray = image.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 각도 예측 변환
    pred_x = int((predicted_angles[0].item() + 1) * (size // 2))
    pred_y = int((predicted_angles[1].item() + 1) * (size // 2))
    target_x = int((target_angles[0].item() + 1) * (size // 2))
    target_y = int((target_angles[1].item() + 1) * (size // 2))

    # 공 감지 여부를 기반으로 원 그리기
    if ball_detected[0]:  # 예측된 공 존재 여부 (0.5를 임계값으로)
        img = cv2.circle(img, (pred_x, pred_y), 5, (0, 0, 255), -1)  # 예측된 좌표 (빨간색)

    # 실제 좌표 그리기
    if ball_detected[1]:
        img = cv2.circle(img, (target_x, target_y), 5, (0, 255, 0), -1)  # 실제 좌표 (초록색)

    # 저장 경로 생성 및 이미지 저장
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, f"result_{idx}.png")
    cv2.imwrite(img_path, img)

if __name__ == "__main__":
    train_csv = './data/train_labels.csv'
    train_folder = './data/train/balls'
    empty_folder = './data/train/empty'
    test_csv = './data/test_labels.csv'
    test_folder = './data/test/balls'
    test_empty_folder = './data/test/empty'

    train_images, train_labels = load_labeled_data(train_csv, train_folder, empty_folder)
    test_images, test_labels = load_labeled_data(test_csv, test_folder, test_empty_folder)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CameraControlDataset(train_images, train_labels, transform=transform)
    test_dataset = CameraControlDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} for training.")

    model = CameraControlNet()
    train_camera_control_model(model, train_loader, val_loader, device, epochs=50, model_save_path='./models')