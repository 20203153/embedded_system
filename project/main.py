import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    Compose, HorizontalFlip, KeypointParams, Normalize, VerticalFlip, RandomBrightnessContrast,
    HueSaturationValue, ShiftScaleRotate, MotionBlur, GaussianBlur,
    GridDistortion, OpticalDistortion, ElasticTransform, CoarseDropout,
    MedianBlur, ColorJitter, Resize
)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random

# ----------------------------
# Heatmap 생성 함수
# ----------------------------
def generate_heatmap(image_size, keypoint, sigma=2):
    height, width = image_size
    y, x = keypoint
    heatmap = np.zeros((height, width), dtype=np.float32)

    if x < 0 or y < 0 or x >= width or y >= height:
        return heatmap

    size = int(6 * sigma + 1)
    x0 = int(x)
    y0 = int(y)

    x_min = max(0, x0 - size // 2)
    x_max = min(width, x0 + size // 2 + 1)
    y_min = max(0, y0 - size // 2)
    y_max = min(height, y0 + size // 2 + 1)

    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    gaussian = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], gaussian)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

class TennisBallDataset(Dataset):
    def __init__(self, images, labels, transform=None, heatmap_size=(240, 320),
                 num_keypoints=1, sigma=2, augmentation_factor=5,
                 original_image_size=(480, 640), margin=6):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.sigma = sigma
        self.augmentation_factor = augmentation_factor
        self.original_image_size = original_image_size
        self.margin = margin

    def __len__(self):
        return len(self.images) * self.augmentation_factor

    def generate_heatmaps(self, keypoints):
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        for i, kp in enumerate(keypoints[:self.num_keypoints]):
            x, y = kp
            heatmaps[i] = generate_heatmap((self.heatmap_size[0], self.heatmap_size[1]), (y, x), self.sigma)
        return heatmaps

    def __getitem__(self, idx):
        real_idx = idx // self.augmentation_factor
        image_path = self.images[real_idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[real_idx]
        x, y, visible = label
        keypoints = [[x, y]] if visible == 1 else []

        # Perform the augmentations
        if self.transform:
            augmented = self.transform(image=image, keypoints=keypoints)
            image = augmented['image']
            keypoints_aug = augmented['keypoints']
            # print(f"Augmented keypoints: {keypoints_aug}")  # Debug statement
        else:
            keypoints_aug = keypoints

        # Heatmap scaling logic
        scale_x = self.heatmap_size[1] / self.original_image_size[1]
        scale_y = self.heatmap_size[0] / self.original_image_size[0]

        if len(keypoints_aug) == 0:
            keypoints_scaled = [[0.0, 0.0]]
            visible = 0
        else:
            kp = keypoints_aug[0]
            x_scaled = kp[0] * scale_x
            y_scaled = kp[1] * scale_y
            x_scaled = np.clip(x_scaled, self.margin, self.heatmap_size[1] - 1 - self.margin)
            y_scaled = np.clip(y_scaled, self.margin, self.heatmap_size[0] - 1 - self.margin)
            keypoints_scaled = [[x_scaled, y_scaled]]
            visible = 1

        heatmaps = self.generate_heatmaps(keypoints_scaled)

        # **Fixed: Scaling the Augmented Keypoints Instead of Original**
        if visible == 1:
            true_x_scaled = kp[0] * scale_x  # Use augmented keypoint
            true_y_scaled = kp[1] * scale_y  # Use augmented keypoint
            true_x_scaled = np.clip(true_x_scaled, self.margin, self.heatmap_size[1] - 1 - self.margin)
            true_y_scaled = np.clip(true_y_scaled, self.margin, self.heatmap_size[0] - 1 - self.margin)
            dataset_label = [true_x_scaled, true_y_scaled, visible]
        else:
            dataset_label = [0.0, 0.0, 0]

        return image, torch.tensor(heatmaps, dtype=torch.float32), dataset_label
    
# ----------------------------
# 모델 정의
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut connection을 위한 컨볼루션 레이어
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()  # 입력 차원과 출력 차원이 같을 때

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # Skip connection
        out = self.activation(out)
        return out

class HeatmapModel(nn.Module):
    def __init__(self, num_keypoints=1, heatmap_size=(240, 320), pretrained=False):
        super(HeatmapModel, self).__init__()
        self.res_blocks = nn.Sequential(
            ResBlock(in_channels=3, out_channels=16),
            ResBlock(in_channels=16, out_channels=16),
            ResBlock(in_channels=16, out_channels=16)
        )
        self.output_heatmaps = nn.Conv2d(16, num_keypoints, kernel_size=1)
        self.heatmap_size = heatmap_size  # (height, width)

    def forward(self, x):
        x = self.res_blocks(x)
        heatmaps = self.output_heatmaps(x)
        heatmaps = nn.functional.interpolate(heatmaps, size=(self.heatmap_size[0], self.heatmap_size[1]), mode='bilinear', align_corners=False)
        return torch.sigmoid(heatmaps)  # 0~1 사이의 값으로 정규화

# ----------------------------
# 조기 종료 클래스 정의
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """손실을 기반으로 조기 종료를 판단"""
        score = -val_loss  # 손실은 최소화하므로 음수로 변환
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """최적 모델을 저장"""
        if self.verbose:
            print(f'Validation loss decreased ({val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), self.path)

# ----------------------------
# 학습 함수 정의
# ----------------------------
def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-4, patience=10, model_save_path='best_model.pth'):
    """
    모델 학습 함수
    Args:
        model (nn.Module): 학습할 모델.
        train_loader (DataLoader): 학습 데이터 로더.
        val_loader (DataLoader): 검증 데이터 로더.
        device (torch.device): 학습 장치.
        epochs (int): 최대 에폭 수.
        lr (float): 학습률.
        patience (int): 조기 종료 인내심.
        model_save_path (str): 최적 모델 저장 경로.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience//2)
    criterion = nn.BCELoss()  # BCE Loss 사용
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    for epoch in tqdm(range(1, epochs + 1), "Epoch", leave=True, mininterval=20):
        model.train()
        running_loss = 0.0
        for images, heatmaps, _ in tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False, mininterval=0.5):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            optimizer.zero_grad()
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, heatmaps)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps, _ in tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False, mininterval=0.5):
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item() * images.size(0)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch}: Train Loss = {epoch_loss:.6f}, Val Loss = {val_epoch_loss:.6f}")
        scheduler.step(val_epoch_loss)
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    # 최적 모델 로드
    model.load_state_dict(torch.load(model_save_path))
    print(f"Best model loaded from {model_save_path}")

# ----------------------------
# 평가 함수 정의
# ----------------------------
def evaluate_model(model, data_loader, device):
    """
    모델 평가 함수
    Args:
        model (nn.Module): 평가할 모델.
        data_loader (DataLoader): 평가 데이터 로더.
        device (torch.device): 평가 장치.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, heatmaps, _ in tqdm(data_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(heatmaps.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mse = mean_squared_error(all_labels.flatten(), all_preds.flatten())
    print(f"Evaluation MSE: {mse:.6f}")
    # 추가적인 평가 지표 (Precision, Recall 등)
    # 여기서는 단일 키포인트이므로 간단히 최대값 위치 비교
    precision = 0.0
    recall = 0.0
    for i in range(len(all_preds)):
        pred_heatmap = all_preds[i][0]
        true_heatmap = all_labels[i][0]
        pred_y, pred_x = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        true_y, true_x = np.unravel_index(np.argmax(true_heatmap), true_heatmap.shape)
        # 예측과 실제 키포인트 간의 거리 계산 (Threshold 설정)
        distance = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        threshold = 2  # 히트맵 좌표 기준
        if distance <= threshold:
            precision += 1
            recall += 1
    precision /= len(all_preds)
    recall /= len(all_preds)
    print(f"Evaluation Precision: {precision:.6f}, Recall: {recall:.6f}")
    return mse, precision, recall

# ----------------------------
# 시각화 함수 수정
# ----------------------------
def check_ball_recognition(heatmap, threshold=0.7):
    """
    Check if the ball is recognized in the heatmap based on a given threshold.

    Args:
        heatmap (numpy.ndarray): The predicted heatmap of shape (H, W).
        threshold (float): The threshold value for determining recognition.

    Returns:
        bool: True if the ball is recognized, False otherwise.
    """
    # Get the maximum value in the heatmap
    max_value = np.max(heatmap)

    # Determine if the maximum value exceeds the threshold
    return max_value >= threshold

def visualize_predictions(model, dataset, device, num_samples=5, heatmap_threshold=0.5):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    scale_factor_x = dataset.original_image_size[1] / dataset.heatmap_size[1]
    scale_factor_y = dataset.original_image_size[0] / dataset.heatmap_size[0]

    for idx in indices:
        image, heatmap, dataset_label = dataset[idx]
        input_image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output_heatmap = model(input_image).squeeze(0).cpu().numpy()

        # Check if the ball is recognized based on the heatmap
        ball_recognized = check_ball_recognition(output_heatmap[0], threshold=heatmap_threshold)

        # Get the predicted keypoint position from the heatmap
        pred_y, pred_x = np.unravel_index(np.argmax(output_heatmap[0]), output_heatmap[0].shape)
        pred_x_original = pred_x * scale_factor_x
        pred_y_original = pred_y * scale_factor_y

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        
        # Show predicted label only if recognized
        if ball_recognized:
            plt.scatter(pred_x_original, pred_y_original, c='r', marker='x', s=100, label='Predicted')

        true_x, true_y, visible = dataset_label 
        if visible == 1:
            true_x_original = true_x * scale_factor_x
            true_y_original = true_y * scale_factor_y
            plt.scatter(true_x_original, true_y_original, c='g', marker='o', s=100, label='True')

        plt.title('Image with Keypoints')
        plt.legend()

        # Second subplot: Predicted heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(output_heatmap[0], cmap='hot', interpolation='nearest')
        plt.title('Predicted Heatmap')
        plt.colorbar()
        plt.show()

# ----------------------------
# 데이터 유효성 검사 함수 정의
# ----------------------------
def validate_dataset(images, labels, image_size=(480, 640), margin=6):
    """
    데이터셋의 원본 이미지와 라벨의 유효성을 검사합니다.
    Args:
        images (list): 이미지 파일 경로 리스트.
        labels (list): 각 이미지에 대한 라벨 리스트 [x, y, visible].
        image_size (tuple): 원본 이미지 크기 (height, width).
        margin (int): 키포인트가 경계에서 최소 거리.
    """
    invalid_entries = []
    for i, (img_path, label) in enumerate(zip(images, labels)):
        # 이미지 파일 존재 여부 확인
        if not os.path.isfile(img_path):
            invalid_entries.append((i, "Image file not found", img_path))
            continue
        
        # 라벨 유효성 검사
        x, y, visible = label
        if visible == 1:
            if not (margin <= x < image_size[1] - margin and margin <= y < image_size[0] - margin):
                invalid_entries.append((i, "Invalid coordinates", label))
    # 결과 출력
    if invalid_entries:
        print("Invalid entries found:")
        for entry in invalid_entries:
            print(f"Index {entry[0]}: {entry[1]} - {entry[2]}")
    else:
        print("All entries are valid.")

# ----------------------------
# 랜덤 샘플 표시 함수 정의
# ----------------------------
def display_random_samples(images, labels, num_samples=5, image_size=(480, 640)):
    """
    랜덤으로 선택된 이미지를 표시하고, 해당 라벨을 매칭합니다.
    Args:
        images (list): 이미지 파일 경로 리스트.
        labels (list): 각 이미지에 대한 라벨 리스트 [x, y, visible].
        num_samples (int): 표시할 샘플 수.
        image_size (tuple): 원본 이미지 크기 (height, width).
    """
    # 랜덤하게 인덱스 선택
    indices = random.sample(range(len(images)), num_samples)
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        img_path = images[idx]
        label = labels[idx]
        x, y, visible = label
        
        # 이미지 불러오기
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)  # 표시할 크기로 조정
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(f"Visible: {visible}")  # Title에 visible 상태만 표시
        # 라벨 위치 마킹
        if visible == 1:
            plt.scatter(x * (image_size[1] / 640), 
                        y * (image_size[0] / 480), 
                        c='g', marker='o', s=100, label='True Label')
            plt.text(x * (image_size[1] / 640), 
                     y * (image_size[0] / 480), 
                     f'({x}, {y})', color='g', fontsize=12, ha='right')
    plt.tight_layout()
    plt.show()

# ----------------------------
# 데이터 로딩 함수 정의
# ----------------------------
def load_data(csv_path, ball_dir, empty_dir, image_size=(480, 640), margin=6):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    for _, row in df.iterrows():
        img_name = row['image']
        x, y = row['x'], row['y']
        img_path = os.path.join(ball_dir, img_name)
        
        if os.path.exists(img_path):
            if margin <= x < image_size[1] - margin and margin <= y < image_size[0] - margin:
                images.append(img_path)
                labels.append([x, y, 1])  # visible = 1
            else:
                print(f"Invalid keypoint: {img_name}, x={x}, y={y}, skipped.")
        else:
            print(f"Image file not found: {img_path}, skipped.")

    for img_name in os.listdir(empty_dir):
        img_path = os.path.join(empty_dir, img_name)
        if os.path.isfile(img_path) and img_name != ".DS_Store":
            images.append(img_path)
            labels.append([0.0, 0.0, 0])  # visible = 0
    return images, labels

# ----------------------------
# 메인 실행 부분
# ----------------------------
if __name__ == "__main__":
    # 데이터 경로 설정
    train_csv = './data/train_labels.csv'
    train_image_dir = './data/train/balls'
    empty_image_dir = './data/train/empty'
    val_csv = './data/test_labels.csv'
    val_image_dir = './data/test/balls'
    val_empty_image_dir = './data/test/empty'
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 학습 데이터 로드
    train_images, train_labels = load_data(train_csv, train_image_dir, empty_image_dir, image_size=(480, 640), margin=6)
    print(f"Loaded {len(train_images)} training samples.")
    
    # 검증 데이터 로드
    val_images, val_labels = load_data(val_csv, val_image_dir, val_empty_image_dir, image_size=(480, 640), margin=6)
    print(f"Loaded {len(val_images)} validation samples.")
    
    # 데이터셋 유효성 검사
    print("Validating training dataset...")
    validate_dataset(train_images, train_labels, image_size=(480, 640), margin=6)
    print("Validating validation dataset...")
    validate_dataset(val_images, val_labels, image_size=(480, 640), margin=6)
    
    # 데이터 증강 및 전처리 파이프라인 설정
    train_transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=0, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        GridDistortion(p=0.2),
        OpticalDistortion(p=0.2),
        ElasticTransform(alpha=1, sigma=50, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        MotionBlur(p=0.2),
        GaussianBlur(p=0.2),
        MedianBlur(blur_limit=3, p=0.1),
        CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.5),
        Resize(height=480, width=640, p=1.0),  # 모든 변환 후 크기 고정
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=KeypointParams(format='xy', remove_invisible=True))
    
    val_transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=0, p=0.5, border_mode=cv2.BORDER_REFLECT),
        GridDistortion(p=0.1),
        OpticalDistortion(p=0.1),
        ElasticTransform(alpha=1, sigma=50, p=0.3),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
        MotionBlur(p=0.1),
        GaussianBlur(p=0.1),
        CoarseDropout(max_holes=2, max_height=8, max_width=8, fill_value=0, p=0.3),
        Resize(height=480, width=640, p=1.0),  # 모든 변환 후 크기 고정
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=KeypointParams(format='xy', remove_invisible=True))
    
    # 데이터셋 생성
    train_dataset = TennisBallDataset(
        images=train_images,
        labels=train_labels,
        transform=train_transform,
        heatmap_size=(240, 320),  # heatmap은 이미지의 절반 크기
        num_keypoints=1,
        sigma=2,
        augmentation_factor=5,  # 각 이미지를 5번 증강
        original_image_size=(480, 640),  # (height, width)
        margin=6  # 마진 설정
    )
    
    val_dataset = TennisBallDataset(
        images=val_images,
        labels=val_labels,
        transform=val_transform,
        heatmap_size=(240, 320),  # heatmap은 이미지의 절반 크기
        num_keypoints=1,
        sigma=2,
        augmentation_factor=3,  # 각 이미지를 3번 증강 (선택사항)
        original_image_size=(480, 640),  # (height, width)
        margin=6  # 마진 설정
    )
    
    # 데이터 로더 설정
    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # 시스템에 맞게 조정 (예: CPU 코어 수)
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,  # 시스템에 맞게 조정 (예: CPU 코어 수)
        pin_memory=True
    )
    
    # 모델 초기화
    model = HeatmapModel(num_keypoints=1, heatmap_size=(240, 320), pretrained=False)
    
    # 학습 수행
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=500,
        lr=1e-3,
        patience=5,
        model_save_path='best_model.pth'
    )

    # 실제 입력 이미지를 사용하여 ONNX 형식으로 모델 저장
    # test_loader에서 이미지 가져오기
    images = []

    # Sample images from the train_dataset
    for i, (image, heatmap, label) in enumerate(train_dataset):
        # Convert image tensor to numpy and append
        images.append(image.permute(1, 2, 0).cpu().numpy())  # Ensure proper channel arrangement (H, W, C)

    # Proceed if there are available images
    if images:
        # Convert list of images to numpy array (N, H, W, C)
        input_tensor = np.array(images)

        # Select a sample of 200 images, if less than 200, adjust accordingly
        if len(input_tensor) > 200:
            idx = np.random.choice(len(input_tensor), 200, replace=False)
            input_tensor = input_tensor[idx]
        
        # Save the input tensor in npy format for potential future use
        input_tensor_npy = input_tensor / 255.0  # Scale input appropriately
        np.save("./input_tensor.npy", input_tensor_npy)

        # Convert numpy array to PyTorch tensor
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)

        # Transpose the input tensor to (N, C, H, W) for PyTorch models
        input_tensor = input_tensor.permute(0, 3, 1, 2)

        # Model export to ONNX format
        onnx_file_path = "model.onnx"
        torch.onnx.export(
            model,
            input_tensor,
            onnx_file_path,
            export_params=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("Model has been exported to ONNX format using images from train_dataset.")
    else:
        print("No valid images available for ONNX export.")

    # Evaluate the model
    mse, precision, recall = evaluate_model(model, val_loader, device)
    print(f"Model Evaluation - MSE: {mse}, Precision: {precision}, Recall: {recall}")

    # Visualization function
    visualize_predictions(model, val_dataset, device, num_samples=15, heatmap_threshold=0.3)

    thresholds = [0.3, 0.35, 0.4, 0.45]
    results = {th: {"True Positives": 0, "True Negatives": 0, "False Positives": 0, "False Negatives": 0} for th in thresholds}
    # test_loader에서 데이터 가져오기
    for i, (image, heatmap, label) in enumerate(val_dataset):
        input_image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output_heatmap = model(input_image).squeeze(0).cpu().numpy()
        
        for th in thresholds:
            ball_recognized = check_ball_recognition(output_heatmap[0], threshold=th)
            if ball_recognized and label[2] == 1:  # if detected and should be visible
                results[th]["True Positives"] += 1
            elif not ball_recognized and label[2] == 1:  # if not detected but should be visible
                results[th]["False Negatives"] += 1
            elif ball_recognized and label[2] == 0:  # if detected but should not be visible
                results[th]["False Positives"] += 1
            else:
                results[th]["True Negatives"] += 1

    # Print results
    for th, metrics in results.items():
        print(f"Threshold {th}: TP={metrics['True Positives']}, TN={metrics['True Negatives']}, FP={metrics['False Positives']}, FN={metrics['False Negatives']}")
    