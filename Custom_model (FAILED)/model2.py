# Full YOLOv8-style Crater Detection Pipeline with Updated Crater Detector and Stability Fixes

# === CONFIG ===
import os
import torch
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 416
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VALID_IMG_DIR = os.path.join(DATASET_DIR, "valid", "images")
VALID_LABEL_DIR = os.path.join(DATASET_DIR, "valid", "labels")

EPOCHS = 60
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CONF_THRESH = 0.45
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "improved_crater_model.pt")

# === Dataset ===
class CraterDataset(Dataset):
    def __init__(self, image_dir, label_dir, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        if self.augment and random.random() > 0.5:
            image = cv2.flip(image, 1)

        image = image.astype(np.float32) / 255.0
        image = np.clip(image * 1.2 - 0.1, 0, 1)
        image = torch.tensor(image).unsqueeze(0).float()

        label_path = os.path.join(self.label_dir, os.path.splitext(img_filename)[0] + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        try:
                            _, x, y, w, h = parts[:5]
                            x = max(0, min(1, x))
                            y = max(0, min(1, y))
                            w = max(0.01, min(1, w))
                            h = max(0.01, min(1, h))
                            boxes.append([x, y, w, h])
                        except ValueError:
                            print(f"Warning: Invalid label format in {label_path}")

        boxes = torch.tensor(boxes).float() if boxes else torch.zeros((0, 4))
        return image, boxes

# === Collate Function ===
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

# === YOLO-style Model Definition ===
class ImprovedCraterDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        self.detector = nn.Conv2d(256, 5, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        return x.permute(0, 2, 3, 1)

# === Corrected YOLO-style Loss Function ===
# === Corrected YOLO-style Loss Function with Final Objectness Fix ===
def yolo_loss(preds, targets, device):
    B, GY, GX, _ = preds.shape
    lambda_coord = 5
    lambda_noobj = 0.5

    obj_mask = torch.zeros((B, GY, GX), device=device)
    noobj_mask = torch.ones((B, GY, GX), device=device)
    tx = torch.zeros((B, GY, GX), device=device)
    ty = torch.zeros((B, GY, GX), device=device)
    tw = torch.zeros((B, GY, GX), device=device)
    th = torch.zeros((B, GY, GX), device=device)

    for b in range(B):
        for box in targets[b]:
            if len(box) < 4:
                continue
            gx, gy, gw, gh = box
            gi = min(GX - 1, max(0, int(gx * GX)))
            gj = min(GY - 1, max(0, int(gy * GY)))

            obj_mask[b, gj, gi] = 1
            noobj_mask[b, gj, gi] = 0
            tx[b, gj, gi] = gx * GX - gi
            ty[b, gj, gi] = gy * GY - gj
            tw[b, gj, gi] = torch.log((gw * GX).clamp(min=1e-8))
            th[b, gj, gi] = torch.log((gh * GY).clamp(min=1e-8))

    pred_conf = torch.sigmoid(preds[..., 0])
    pred_x = torch.sigmoid(preds[..., 1])
    pred_y = torch.sigmoid(preds[..., 2])
    pred_w = preds[..., 3]
    pred_h = preds[..., 4]

    tconf = obj_mask.float()

    if obj_mask.sum() > 0:
        coord_loss = lambda_coord * (
            F.mse_loss(pred_x[obj_mask == 1], tx[obj_mask == 1], reduction='sum') +
            F.mse_loss(pred_y[obj_mask == 1], ty[obj_mask == 1], reduction='sum') +
            F.mse_loss(pred_w[obj_mask == 1], tw[obj_mask == 1], reduction='sum') +
            F.mse_loss(pred_h[obj_mask == 1], th[obj_mask == 1], reduction='sum')
        ) / B
    else:
        coord_loss = torch.tensor(0.0, device=device)

    # Compute objectness loss over all grid cells
    obj_loss = F.binary_cross_entropy(pred_conf, tconf)

    total_loss = coord_loss + obj_loss
    return total_loss

# === Training Functions ===
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for imgs, targets in tqdm(dataloader, desc="Training"):
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = yolo_loss(outputs, targets, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0

    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(imgs)
        loss = yolo_loss(outputs, targets, device)
        total_loss += loss.item()

    return total_loss / len(dataloader)

# === Main Training Loop ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CraterDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=True)
    valid_dataset = CraterDataset(VALID_IMG_DIR, VALID_LABEL_DIR, augment=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ImprovedCraterDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    patience = 0
    max_patience = 5

    for epoch in range(EPOCHS):
        print(f"\n--- EPOCH {epoch+1}/{EPOCHS} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, valid_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Best model saved!")
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f"Early stopping after {max_patience} epochs without improvement")
            break

    print("Training completed!")
