import os
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

class CameraControlNet(nn.Module):
    def __init__(self):
        super(CameraControlNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # (256, 256, 3) -> (256, 256, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (256, 256, 32) -> (128, 128, 32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # (128, 128, 32) -> (128, 128, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 128, 64) -> (64, 64, 64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # (64, 64, 64) -> (64, 64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 64, 128) -> (32, 32, 128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # (32, 32, 128) -> (32, 32, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 32, 256) -> (16, 16, 256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)  # 256 * 16 * 16 = 65536
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 2)  # Output: 2 angles (x, y)

        # Activation functions and dropout
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))  # First convolution layer
        x = self.pool2(self.relu(self.conv2(x)))  # Second convolution layer
        x = self.pool3(self.relu(self.conv3(x)))  # Third convolution layer
        x = self.pool4(self.relu(self.conv4(x)))  # Fourth convolution layer

        x = x.view(-1, 256 * 16 * 16)  # Flatten the tensor for fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))  # First fully connected layer
        x = self.dropout(self.relu(self.fc2(x)))  # Second fully connected layer
        x = torch.tanh(self.fc_out(x))  # Output layer with tanh to keep output between -1 and 1

        return x

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
        labels.append([norm_x, norm_y])

    for empty_file in os.listdir(empty_folder):
        empty_img_path = str(os.path.join(empty_folder, empty_file))
        img = cv2.imread(empty_img_path, cv2.IMREAD_COLOR)  
        if img is None:
            print(f"Unable to load image: {empty_img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        labels.append([0, 0])  

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
    criterion = nn.MSELoss()

    os.makedirs(model_save_path, exist_ok=True)  # 모델 저장 경로 생성

    for epoch in tqdm(range(epochs), "Epoch", position=1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, "Batch", position=0):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(model_save_path, "fine_tuned_model.pth"))
    print(f"Model saved at {model_save_path}")

    for idx, (images, labels) in tqdm(enumerate(val_loader), "Gen image"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        for i in range(images.size(0)):
            save_result_image(images[i], outputs[i], labels[i], idx * len(images) + i, save_dir=results_dir)

    onnx_path = os.path.join(model_save_path, 'camera_control_model.onnx')
    sample_img, _ = next(iter(train_loader))
    sample_img = sample_img[0:1].to(device)

    torch.onnx.export(model, sample_img, onnx_path, opset_version=11)
    print(f"Model saved as ONNX format at: {onnx_path}")

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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} for training.")

    model = CameraControlNet()
    train_camera_control_model(model, train_loader, val_loader, device, epochs=50, model_save_path='./models')