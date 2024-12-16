import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import (Compose, HorizontalFlip, KeypointParams, Normalize, VerticalFlip, 
                            RandomBrightnessContrast, HueSaturationValue, ShiftScaleRotate, 
                            MotionBlur, GaussianBlur, GridDistortion, OpticalDistortion, 
                            ElasticTransform, CoarseDropout, MedianBlur, ColorJitter, Resize)
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
    x0, y0 = int(x), int(y)
    x_min, x_max = max(0, x0 - size // 2), min(width, x0 + size // 2 + 1)
    y_min, y_max = max(0, y0 - size // 2), min(height, y0 + size // 2 + 1)
    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    gaussian = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], gaussian)
    return heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

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
        else:
            keypoints_aug = keypoints
        
        # Heatmap scaling logic
        scale_x = self.heatmap_size[1] / self.original_image_size[1]
        scale_y = self.heatmap_size[0] / self.original_image_size[0]

        if len(keypoints_aug) == 0:
            keypoints_scaled = [[0.0, 0.0]]
            visible = 0
            x_scaled, y_scaled = 0.0, 0.0
        else:
            kp = keypoints_aug[0]
            x_scaled = np.clip(kp[0] * scale_x, self.margin, self.heatmap_size[1] - 1 - self.margin)
            y_scaled = np.clip(kp[1] * scale_y, self.margin, self.heatmap_size[0] - 1 - self.margin)
            keypoints_scaled = [[x_scaled, y_scaled]]
            visible = 1
            
            # Alternatively, include true coordinates directly for the dataset label
            true_x_scaled = kp[0] * scale_x  
            true_y_scaled = kp[1] * scale_y  
            true_x_scaled = np.clip(true_x_scaled, self.margin, self.heatmap_size[1] - 1 - self.margin)
            true_y_scaled = np.clip(true_y_scaled, self.margin, self.heatmap_size[0] - 1 - self.margin)
        
        # Generate heatmaps using scaled keypoints
        heatmaps = self.generate_heatmaps(keypoints_scaled)
        
        if visible == 1:
            dataset_label = [true_x_scaled, true_y_scaled, visible]
        else:
            dataset_label = [0.0, 0.0, 0]

        return image, torch.tensor(heatmaps, dtype=torch.float32), dataset_label

# ----------------------------
# Model Definitions
# ----------------------------

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return self.activation(out)

class HeatmapModel(nn.Module):
    def __init__(self, num_keypoints=1, heatmap_size=(240, 320), pretrained=False):
        super(HeatmapModel, self).__init__()
        self.res_blocks = nn.Sequential(*[ResBlock(3 if i == 0 else 16, 16) for i in range(32)])  # 10 residual blocks
        self.output_heatmaps = nn.Conv2d(16, num_keypoints, kernel_size=1)
        self.heatmap_size = heatmap_size

    def forward(self, x):
        x = self.res_blocks(x)
        heatmaps = self.output_heatmaps(x)
        return torch.sigmoid(nn.functional.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False))

# ----------------------------
# Early Stopping class
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
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# ----------------------------
# Training Function
# ----------------------------

def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-4, patience=10, model_save_path='best_model.pth', delta=0.0):
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience // 2)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path, delta=delta)

    for epoch in tqdm(range(1, epochs + 1), "Epoch", leave=True, mininterval=20):
        model.train()
        running_loss = 0.0
        for images, heatmaps, _ in tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False, mininterval=0.5):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
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

    # Load the best model
    model.load_state_dict(torch.load(model_save_path))
    print(f"Best model loaded from {model_save_path}")

# ----------------------------
# Evaluation Function
# ----------------------------

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []

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

    # Further evaluation metrics (Precision, Recall)
    precision, recall = 0.0, 0.0
    for i in range(len(all_preds)):
        pred_heatmap = all_preds[i][0]
        true_heatmap = all_labels[i][0]
        pred_y, pred_x = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        true_y, true_x = np.unravel_index(np.argmax(true_heatmap), true_heatmap.shape)

        distance = np.sqrt((pred_x - true_x) **2 + (pred_y - true_y)** 2)
        threshold = 2  # Heatmap coordinate threshold
        if distance <= threshold:
            precision += 1
            recall += 1

    precision /= len(all_preds)
    recall /= len(all_preds)
    print(f"Evaluation Precision: {precision:.6f}, Recall: {recall:.6f}")

    return mse, precision, recall

# ----------------------------
# Check Ball Recognition Function
# ----------------------------

def check_ball_recognition(heatmap, threshold=0.7):
    max_value = np.max(heatmap)
    return max_value >= threshold

# ----------------------------
# Visualization Function
# ----------------------------

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

        ball_recognized = check_ball_recognition(output_heatmap[0], threshold=heatmap_threshold)
        pred_y, pred_x = np.unravel_index(np.argmax(output_heatmap[0]), output_heatmap[0].shape)
        pred_x_original = pred_x * scale_factor_x
        pred_y_original = pred_y * scale_factor_y

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        
        if ball_recognized:
            plt.scatter(pred_x_original, pred_y_original, c='r', marker='x', s=100, label='Predicted')
        
        true_x, true_y, visible = dataset_label
        if visible == 1:
            true_x_original = true_x * scale_factor_x
            true_y_original = true_y * scale_factor_y
            plt.scatter(true_x_original, true_y_original, c='g', marker='o', s=100, label='True')
        
        plt.title('Image with Keypoints')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.imshow(output_heatmap[0], cmap='hot', interpolation='nearest')
        plt.title('Predicted Heatmap')
        plt.colorbar()
        #plt.show()
        plt.savefig(f"{idx + 1}_preds.png")

# ----------------------------
# Dataset Validation Function
# ----------------------------

def validate_dataset(images, labels, image_size=(480, 640), margin=6):
    invalid_entries = []
    for i, (img_path, label) in enumerate(zip(images, labels)):
        if not os.path.isfile(img_path):
            invalid_entries.append((i, "Image file not found", img_path))
            continue
        x, y, visible = label
        if visible == 1:
            if not (margin <= x < image_size[1] - margin and margin <= y < image_size[0] - margin):
                invalid_entries.append((i, "Invalid coordinates", label))
    if invalid_entries:
        print("Invalid entries found:")
        for entry in invalid_entries:
            print(f"Index {entry[0]}: {entry[1]} - {entry[2]}")
    else:
        print("All entries are valid.")

# ----------------------------
# Display Random Samples Function
# ----------------------------

def display_random_samples(images, labels, num_samples=5, image_size=(480, 640)):
    indices = random.sample(range(len(images)), num_samples)
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        img_path = images[idx]
        label = labels[idx]
        x, y, visible = label
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(f"Visible: {visible}")
        
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
# Load Data Function
# ----------------------------

def load_data(csv_path, ball_dir, empty_dir, image_size=(480, 640), margin=6):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    # Load images and labels for the 'balls' directory
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

    # Load images and labels for the 'empty' directory
    for img_name in os.listdir(empty_dir):
        img_path = os.path.join(empty_dir, img_name)
        if os.path.isfile(img_path) and img_name != ".DS_Store":
            images.append(img_path)
            labels.append([0.0, 0.0, 0])  # visible = 0
    
    # Combine images and labels, then shuffle
    combined = list(zip(images, labels))
    random.shuffle(combined)
    shuffled_images, shuffled_labels = zip(*combined)  # Unzip

    return list(shuffled_images), list(shuffled_labels)

# ----------------------------
# Main Execution Block
# ----------------------------
if __name__ == "__main__":
    train_csv = './data/train_labels.csv'
    train_image_dir = './data/train/balls'
    empty_image_dir = './data/train/empty'
    val_csv = './data/test_labels.csv'
    val_image_dir = './data/test/balls'
    val_empty_image_dir = './data/test/empty'
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load datasets
    train_images, train_labels = load_data(train_csv, train_image_dir, empty_image_dir, image_size=(480, 640), margin=6)
    print(f"Loaded {len(train_images)} training samples.")
    
    val_images, val_labels = load_data(val_csv, val_image_dir, val_empty_image_dir, image_size=(480, 640), margin=6)
    print(f"Loaded {len(val_images)} validation samples.")
    
    # Validate datasets
    print("Validating training dataset...")
    validate_dataset(train_images, train_labels, image_size=(480, 640), margin=6)
    print("Validating validation dataset...")
    validate_dataset(val_images, val_labels, image_size=(480, 640), margin=6)
    
    # Define data augmentation
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
        Resize(height=480, width=640, p=1.0),
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
        Resize(height=480, width=640, p=1.0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=KeypointParams(format='xy', remove_invisible=True))

    # Create datasets
    train_dataset = TennisBallDataset(
        images=train_images,
        labels=train_labels,
        transform=train_transform,
        heatmap_size=(240, 320),
        num_keypoints=1,
        sigma=2,
        augmentation_factor=5,
        original_image_size=(480, 640),
        margin=6
    )
    
    # Splitting Train Dataset
    train_len = int(len(train_dataset) * 0.8)
    train_sets = torch.utils.data.Subset(train_dataset, range(0, train_len))
    val_sets = torch.utils.data.Subset(train_dataset, range(train_len, len(train_dataset)))

    test_dataset = TennisBallDataset(
        images=val_images,
        labels=val_labels,
        transform=val_transform,
        heatmap_size=(240, 320),
        num_keypoints=1,
        sigma=2,
        augmentation_factor=3,
        original_image_size=(480, 640),
        margin=6
    )
    
    # DataLoader Setup
    batch_size = 16 
    train_loader = DataLoader(
        train_sets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_sets,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    # Initialize the model
    model = HeatmapModel(num_keypoints=1, heatmap_size=(240, 320), pretrained=False)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=100,
        lr=1e-3,
        patience=5,
        model_save_path='best_model.pth',
        delta=1e-5
    )
    
    # Export model to ONNX
    def image_generator(val_dataset):
        """Generator to yield processed images one at a time to save memory."""
        for image, _, _ in val_dataset:
            yield image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)

    # Collect images using a generator
    images = []
    for image in image_generator(train_dataset):
        images.append(image)
        if len(images) == 50:  # Limit to the first 200 images
            break

    # Proceed if there are valid images
    if images:
        input_tensor = np.array(images, dtype=np.float32)  # Specify dtype to save memory
        input_tensor /= 255.0  # Normalize inputs
        np.save("./input_tensor.npy", input_tensor)
        
        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)
        input_tensor = input_tensor.permute(0, 3, 1, 2)  # Reshape to (N, C, H, W)

        # Export model to ONNX format
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
    mse, precision, recall = evaluate_model(model, test_loader, device)
    print(f"Model Evaluation - MSE: {mse}, Precision: {precision}, Recall: {recall}")

    # Visualize predictions
    visualize_predictions(model, test_dataset, device, num_samples=15, heatmap_threshold=0.5)
    
    thresholds = [0.3, 0.4, 0.5, 0.6]
    results = {th: {"True Positives": 0, "True Negatives": 0, "False Positives": 0, "False Negatives": 0} for th in thresholds}
    
    # Evaluate on test dataset
    for i, (image, heatmap, label) in enumerate(test_dataset):
        input_image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output_heatmap = model(input_image).squeeze(0).cpu().numpy()
        for th in thresholds:
            ball_recognized = check_ball_recognition(output_heatmap[0], threshold=th)
            if ball_recognized and label[2] == 1:  # If detected and should be visible
                results[th]["True Positives"] += 1
            elif not ball_recognized and label[2] == 1:  # If not detected but should be visible
                results[th]["False Negatives"] += 1
            elif ball_recognized and label[2] == 0:  # If detected but should not be visible
                results[th]["False Positives"] += 1
            else:
                results[th]["True Negatives"] += 1

    # Print results
    for th, metrics in results.items():
        print(f"Threshold {th}: TP={metrics['True Positives']}, TN={metrics['True Negatives']}, FP={metrics['False Positives']}, FN={metrics['False Negatives']}")
