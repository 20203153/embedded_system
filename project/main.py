import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, HueSaturationValue, KeypointParams
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

# 데이터셋 클래스 정의
class CameraControlDataset(Dataset):
    def __init__(self, images, labels, transform=None, original_size=(512, 512), augmentation_factor=16):
        self.images = images  # 리스트 형태 유지
        self.labels = labels  # 리스트 형태 유지
        self.transform = transform
        self.original_size = original_size  # 원본 이미지 크기 (512, 512)
        self.augmentation_factor = augmentation_factor

    def __len__(self):
        return len(self.images) * self.augmentation_factor

    def __getitem__(self, idx):
        actual_idx = idx % len(self.images)  # 원본 데이터 인덱스 순환
        image = self.images[actual_idx]
        label = self.labels[actual_idx].copy()  # 라벨의 복사본 사용

        # 공이 있는 경우 키포인트 설정
        if label[2] == 1:
            keypoints = [label[:2]]
        else:
            keypoints = []

        data = {"image": image, "keypoints": keypoints}

        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            keypoints = augmented["keypoints"]

            if len(keypoints) > 0:
                # 키포인트가 있는 경우 라벨 업데이트
                label[:2] = keypoints[0]
                label[2] = 1
            else:
                # 키포인트가 없는 경우 (이미지 밖으로 나간 경우)
                label = [0, 0, 0]
        else:
            if label[2] == 0:
                label = [0, 0, 0]

        # 이미지 크기 얻기 (채널, 높이, 너비)
        _, height, width = image.shape

        # 좌표를 [-1, 1]로 정규화
        norm_x = 2 * (label[0] / width) - 1
        norm_y = 2 * (label[1] / height) - 1
        label = [norm_x, norm_y, label[2]]

        return image, torch.tensor(label, dtype=torch.float32)

# 데이터 로딩 함수
def load_labeled_data(csv_path, image_folder, empty_folder):
    labeled_data = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in labeled_data.iterrows():
        image_file = row['image']
        x, y = row['x'], row['y']

        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB로 변환
        images.append(img)

        # 좌표는 원본 이미지 크기(512x512) 기준
        labels.append([x, y, 1])  # (x, y, has_ball)

    # 공이 없는 이미지 로드
    for empty_file in os.listdir(empty_folder):
        empty_img_path = os.path.join(empty_folder, empty_file)
        img = cv2.imread(empty_img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Unable to load image: {empty_img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append([0, 0, 0])  # (x, y, has_ball=0)

    return images, labels

# 경량화된 모델 정의
class CameraControlNet(nn.Module):
    def __init__(self):
        super(CameraControlNet, self).__init__()

        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)

        # 깊이별 분리 합성곱 블록
        self.blocks = nn.Sequential(
            self._make_dsconv_block(32, 64, stride=1),
            self._make_dsconv_block(64, 128, stride=2),
            self._make_dsconv_block(128, 128, stride=1),
            self._make_dsconv_block(128, 256, stride=2),
            self._make_dsconv_block(256, 256, stride=1),
            self._make_dsconv_block(256, 512, stride=2),
            # 필요한 경우 추가 블록
        )

        # 전역 평균 풀링 및 출력 레이어
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 128)
        self.fc1_act = nn.LeakyReLU(inplace=True)
        self.fc1_dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 32)
        self.fc2_act = nn.LeakyReLU(inplace=True)
        self.fc2_dropout = nn.Dropout(0.2)

        self.output_angles = nn.Linear(32, 2)
        self.output_ball = nn.Linear(32, 1)

    def _make_dsconv_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        x = self.blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1_dropout(self.fc1_act(self.fc1(x)))
        x = self.fc2_dropout(self.fc2_act(self.fc2(x)))
        angles = torch.tanh(self.output_angles(x))  # 좌표를 [-1, 1]로 제한
        ball_detect = self.output_ball(x)  # sigmoid 적용하지 않음
        return angles, ball_detect

# 모델 프루닝 함수
def apply_pruning(model, amount=0.2):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

# 학습 함수
def train_camera_control_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3, model_save_path='./models', results_dir='./results'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    criterion_angle = nn.SmoothL1Loss()
    criterion_ball = nn.BCEWithLogitsLoss()

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    alpha = 10.0  # 좌표 손실의 가중치 증가

    for epoch in tqdm(range(epochs), desc="Epoch", position=0):
        model.train()
        running_loss_angle = 0.0
        running_loss_ball = 0.0
        for images, labels in tqdm(train_loader, desc="Batch", position=1, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs_angles, outputs_ball = model(images)

            # Angles loss
            loss_angle = criterion_angle(outputs_angles, labels[:, :2])

            # Ball detection loss
            loss_ball = criterion_ball(outputs_ball.squeeze(), labels[:, 2])

            # Total loss with weighting
            loss = alpha * loss_angle + loss_ball
            loss.backward()
            optimizer.step()

            running_loss_angle += loss_angle.item()
            running_loss_ball += loss_ball.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss_angle = 0.0
        val_loss_ball = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs_angles, outputs_ball = model(images)

                val_loss_angle += criterion_angle(outputs_angles, labels[:, :2]).item()
                val_loss_ball += criterion_ball(outputs_ball.squeeze(), labels[:, 2]).item()

        avg_train_loss_angle = running_loss_angle / len(train_loader)
        avg_train_loss_ball = running_loss_ball / len(train_loader)
        avg_val_loss_angle = val_loss_angle / len(val_loader)
        avg_val_loss_ball = val_loss_ball / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Angle Loss: {avg_train_loss_angle:.6f}, Train Ball Loss: {avg_train_loss_ball:.6f}, "
              f"Val Angle Loss: {avg_val_loss_angle:.6f}, Val Ball Loss: {avg_val_loss_ball:.6f}, "
              f"LR: {current_lr:.6e}")

    # 모델 저장
    torch.save(model.state_dict(), os.path.join(model_save_path, "camera_control_model.pth"))
    print(f"Model saved at {model_save_path}")


# 예측 결과 시각화 함수
def visualize_predictions(model, dataset, device, num_samples=5):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for idx in indices:
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_angles, pred_ball = model(image)
        pred_angles = pred_angles.squeeze().cpu().numpy()
        pred_ball = torch.sigmoid(pred_ball).item()

        # 좌표 복원
        _, height, width = image.shape[1:]
        pred_x = ((pred_angles[0] + 1) / 2) * width
        pred_y = ((pred_angles[1] + 1) / 2) * height
        true_x = ((label[0].item() + 1) / 2) * width
        true_y = ((label[1].item() + 1) / 2) * height

        # 이미지 시각화
        img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        if label[2].item() == 1:
            plt.scatter(true_x, true_y, c='g', label='True', s=40)
        if pred_ball > 0.5:
            plt.scatter(pred_x, pred_y, c='r', label='Predicted', s=40)
        plt.title(f"Has ball: {label[2].item()}, Predicted has ball: {pred_ball:.2f}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    train_csv = './data/train_labels.csv'
    train_folder = './data/train/balls'
    empty_folder = './data/train/empty'
    test_csv = './data/test_labels.csv'
    test_folder = './data/test/balls'
    test_empty_folder = './data/test/empty'

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} for training.")

    # 원본 이미지 크기 (512, 512)
    original_size = (512, 512)
    # 모델 입력 이미지 크기 (256, 256)
    input_size = (256, 256)

    batch_size = 16

    train_images, train_labels = load_labeled_data(train_csv, train_folder, empty_folder)
    test_images, test_labels = load_labeled_data(test_csv, test_folder, test_empty_folder)

    # 데이터 증강 및 전처리
    train_transform = Compose([
        Resize(*input_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(p=0.5),
        Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ], keypoint_params=KeypointParams(format='xy', remove_invisible=False))

    test_transform = Compose([
        Resize(*input_size),
        Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ], keypoint_params=KeypointParams(format='xy', remove_invisible=False))

    train_dataset = CameraControlDataset(train_images, train_labels, transform=train_transform, original_size=original_size)
    test_dataset = CameraControlDataset(test_images, test_labels, transform=test_transform, original_size=original_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 생성 및 프루닝 적용
    model = CameraControlNet()
    model = apply_pruning(model, amount=0.2)

    # 학습
    train_camera_control_model(model, train_loader, val_loader, device, epochs=50, model_save_path='./models')

    # 양자화 적용
    model.eval()

    # 예측 결과 시각화
    visualize_predictions(model, test_dataset, device, num_samples=5)
