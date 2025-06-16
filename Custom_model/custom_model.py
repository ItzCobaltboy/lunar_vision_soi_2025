import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from torchvision.ops import nms


# This is our custom model for crater detection built from scratch. inspired by YOLO architecture. Builds on with our custom features
# Documentation is provided along side, however wee need to train this model on a large dataset for decent amount of epochs to see outcomes
# We reccommend a Minimum of 600-1000 epochs for decent results.
# We did NOT use any pretrained weights, and we did NOT have time or powerful hardware to test this model
# However we hope it performs asw


####################### CONFIGURATION AND HYPERPARAMETERS #########################
IMG_SIZE = 416 # Dataset images are 416x416 square, no upscaling or downscaling used
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset_sampled")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VALID_IMG_DIR = os.path.join(DATASET_DIR, "valid", "images")
VALID_LABEL_DIR = os.path.join(DATASET_DIR, "valid", "labels")
EVAL_SAVE_PATH = os.path.join(BASE_DIR, "Evalutation_Results")
os.makedirs(EVAL_SAVE_PATH, exist_ok=True)

EPOCHS = 800
BATCH_SIZE = 4  # Reduced for better gradient updates
LEARNING_RATE = 5e-3  # Reduced learning rate
WEIGHT_DECAY = 1e-3  # Added weight decay for regularization
CONF_THRESH = 0.80 # Increased confidence threshold
MODEL_SAVE_PATH_LATEST = os.path.join(BASE_DIR, "custom_model_latest.pt")
MODEL_SAVE_PATH_BEST = os.path.join(BASE_DIR, "custom_model_best.pt")

# Loss weights, these are crucial for balancing the loss components
lambda_coord = 2.0
lambda_noobj = 4.0
lambda_obj = 4.0

################# Main functon at final execution point #################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets with augmentation for training
    train_dataset = CraterDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=True)
    valid_dataset = CraterDataset(VALID_IMG_DIR, VALID_LABEL_DIR, augment=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    # Generate anchors using K-mean
    # Generate anchors
    anchor_generator = KMeansAnchorGenerator(train_dataset, num_anchors=3, img_size=IMG_SIZE)
    ANCHORS = anchor_generator.run_kmeans()
    
    # Create model
    model = ImprovedCraterDetector(num_anchors=3).to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 0
    max_patience = 20

    train_losses = []
    val_losses = []
    obj_losses = []
    noobj_losses = []
    bbox_losses = []

    print("\n=== STARTING TRAINING ===")
    for epoch in range(EPOCHS):
        print(f"\n--- EPOCH {epoch+1}/{EPOCHS} ---")
        
        # Training
        train_loss, obj_loss, noobj_loss, bbox_loss = train_one_epoch(
            model, train_loader, optimizer, device, ANCHORS, epoch)
        
        # Validation
        val_loss = validate_one_epoch(model, valid_loader, device, ANCHORS)
        

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        obj_losses.append(obj_loss)
        noobj_losses.append(noobj_loss)
        bbox_losses.append(bbox_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} (obj: {obj_loss:.4f}, noobj: {noobj_loss:.4f}, bbox: {bbox_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'anchors': ANCHORS
            }, MODEL_SAVE_PATH_LATEST)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'anchors': ANCHORS
            }, MODEL_SAVE_PATH_BEST)
            print("✅ Best model saved!")
            patience = 0
        else:
            patience += 1
            
        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping after {max_patience} epochs without improvement")
            break
    
    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    checkpoint = torch.load(MODEL_SAVE_PATH_LATEST)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    evaluate_dataset_no_iou("Train", model, train_dataset, ANCHORS, device, conf_thresh=CONF_THRESH)
    evaluate_dataset_no_iou("Validation", model, valid_dataset, ANCHORS, device, conf_thresh=CONF_THRESH)

    
    print("Training completed!")

# KMeans Anchor Generator
class KMeansAnchorGenerator:
    def __init__(self, dataset, num_anchors=3, img_size=416):
        self.dataset = dataset
        self.num_anchors = num_anchors
        self.img_size = img_size

    def extract_boxes(self):
        whs = []
        for _, targets in tqdm(self.dataset, desc="Extracting boxes"):
            for box in targets:
                _, _, w, h = box.tolist()
                whs.append([w, h])  # Keep normalized format
        return np.array(whs)

    def run_kmeans(self):
        whs = self.extract_boxes()
        if len(whs) < self.num_anchors:
            # Default anchors if not enough data
            return [[0.1, 0.1], [0.3, 0.3], [0.6, 0.6]]
        
        kmeans = KMeans(n_clusters=self.num_anchors, n_init=10, random_state=42)
        kmeans.fit(whs)
        anchors = kmeans.cluster_centers_.tolist()
        
        # Sort anchors by area (small to large)
        anchors.sort(key=lambda x: x[0] * x[1])
        
        print("📐 Generated Anchors (w, h):")
        for i, (w, h) in enumerate(anchors):
            print(f"Anchor {i+1}: ({w:.4f}, {h:.4f})")
        
        return anchors
        
# === DATASET ===
class CraterDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Simple augmentation
        if self.augment and random.random() > 0.5:
            image = cv2.flip(image, 1)  # Horizontal flip
        
        # Normalize to [0, 1] and add some contrast
        image = image.astype(np.float32) / 255.0
        image = np.clip(image * 1.2 - 0.1, 0, 1)  # Slight contrast enhancement
        
        image = torch.tensor(image).unsqueeze(0).float()  # [1, H, W]

        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(img_filename)[0] + ".txt"
        )

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        _, x, y, w, h = parts[:5]
                        # Clamp values to valid range
                        x = max(0, min(1, x))
                        y = max(0, min(1, y))
                        w = max(0.01, min(1, w))
                        h = max(0.01, min(1, h))
                        boxes.append([x, y, w, h])

        boxes = torch.tensor(boxes).float() if boxes else torch.zeros((0, 4))
        return image, boxes

# === IMPROVED MODEL ===
class ImprovedCraterDetector(nn.Module):
    def __init__(self, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        
        # More sophisticated backbone with residual connections
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        )
        
        # Detection head
        self.detect = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, num_anchors * 5, 1)  # 5 = (x, y, w, h, conf)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.detect(x)  # [B, num_anchors*5, H, W]
        
        B, _, H, W = x.shape
        x = x.view(B, self.num_anchors, 5, H, W).permute(0, 1, 3, 4, 2)  # [B, num_anchors, H, W, 5]
        return x

# === IMPROVED LOSS FUNCTION ===
def bbox_ciou(pred_boxes, target_boxes, eps=1e-7):
    """
    Compute CIoU loss between predicted and target boxes in [x_center, y_center, width, height] format.
    Boxes are in absolute grid-scale (not normalized).
    """
    # Convert to corner format
    p_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    p_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    p_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    p_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    t_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    t_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    t_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    t_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

    # Intersection
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
    area_t = (t_x2 - t_x1) * (t_y2 - t_y1)
    union = area_p + area_t - inter_area + eps
    iou = inter_area / union

    # Center distance
    center_dist = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2 + \
                  (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
    enclose_x1 = torch.min(p_x1, t_x1)
    enclose_y1 = torch.min(p_y1, t_y1)
    enclose_x2 = torch.max(p_x2, t_x2)
    enclose_y2 = torch.max(p_y2, t_y2)
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio consistency
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(target_boxes[..., 2] / (target_boxes[..., 3] + eps)) -
        torch.atan(pred_boxes[..., 2] / (pred_boxes[..., 3] + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / c2 + alpha * v)
    return 1.0 - ciou  # Loss

def improved_yolo_loss(preds, targets, anchors, grid_size=None,
                      lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5):
    """
    Improved YOLO loss function with CIoU and distance-weighted no-object loss.

    Args:
        preds: [B, A, H, W, 5] - Model predictions (from your forward pass)
        targets: List of [N, 4] tensors - Ground truth boxes for each batch
        anchors: List of (w, h) tuples - Anchor boxes
        grid_size: Optional (H, W) - If None, inferred from preds shape
        lambda_coord: Coordinate loss weight
        lambda_obj: Object confidence loss weight
        lambda_noobj: No-object confidence loss weight
    """
    device = preds.device
    B, A, H, W, _ = preds.shape  # Match your forward pass output format

    # Use actual tensor dimensions as grid size
    GY, GX = H, W

    total_loss = 0.0
    total_obj_loss = 0.0
    total_noobj_loss = 0.0
    total_bbox_loss = 0.0

    for b in range(B):
        target = targets[b].to(device) if targets[b] is not None else torch.empty(0, 4, device=device)
        pred = preds[b]  # [A, H, W, 5] - matches your model output

        # Initialize masks and targets using actual grid dimensions
        obj_mask = torch.zeros((A, GY, GX), dtype=torch.bool, device=device)
        noobj_mask = torch.ones((A, GY, GX), dtype=torch.bool, device=device)
        t_box = torch.zeros((A, GY, GX, 4), device=device)
        t_conf = torch.zeros((A, GY, GX), device=device)
        true_centers = []

        # Process each ground truth box
        for box in target:
            if len(box) < 4:
                continue

            cx, cy, w, h = box[:4]

            # Convert to grid coordinates
            gx = cx * GX
            gy = cy * GY
            gi = int(torch.clamp(torch.floor(gx), 0, GX - 1).item())
            gj = int(torch.clamp(torch.floor(gy), 0, GY - 1).item())

            # Create ground truth box in grid coordinates
            gt_box = torch.tensor([gx, gy, w * GX, h * GY], device=device)
            true_centers.append((gx, gy))

            # Find best anchor for this ground truth box
            best_iou = 0
            best_a = 0
            for a in range(A):
                if a >= len(anchors):
                    continue

                aw, ah = anchors[a]
                # Center anchor at grid cell center
                anchor_box = torch.tensor([gi + 0.5, gj + 0.5, aw * GX, ah * GY], device=device)
                iou = calculate_iou(gt_box, anchor_box)
                if iou > best_iou:
                    best_iou = iou
                    best_a = a

            # Assign ground truth to best anchor
            obj_mask[best_a, gj, gi] = True
            noobj_mask[best_a, gj, gi] = False
            t_box[best_a, gj, gi] = gt_box
            t_conf[best_a, gj, gi] = 1.0

        # Decode predictions
        pred_x = torch.sigmoid(pred[..., 0])
        pred_y = torch.sigmoid(pred[..., 1])
        pred_w = pred[..., 2]
        pred_h = pred[..., 3]
        pred_conf = torch.sigmoid(pred[..., 4])

        # Create coordinate grids
        grid_x = torch.arange(GX, device=device, dtype=torch.float32).view(1, 1, GX).expand(A, GY, GX)
        grid_y = torch.arange(GY, device=device, dtype=torch.float32).view(1, GY, 1).expand(A, GY, GX)

        # Ensure we don't go out of bounds with anchors
        anchor_w = torch.zeros(A, device=device)
        anchor_h = torch.zeros(A, device=device)
        for a in range(min(A, len(anchors))):
            anchor_w[a] = anchors[a][0]
            anchor_h[a] = anchors[a][1]

        anchor_w = anchor_w.view(A, 1, 1).expand(A, GY, GX)
        anchor_h = anchor_h.view(A, 1, 1).expand(A, GY, GX)

        # Decode predicted boxes
        pred_boxes = torch.zeros((A, GY, GX, 4), device=device)
        pred_boxes[..., 0] = pred_x + grid_x  # cx
        pred_boxes[..., 1] = pred_y + grid_y  # cy
        pred_boxes[..., 2] = torch.exp(torch.clamp(pred_w, max=10)) * anchor_w  # w (clamp to prevent overflow)
        pred_boxes[..., 3] = torch.exp(torch.clamp(pred_h, max=10)) * anchor_h  # h

        # Calculate losses
        bbox_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)

        # CIoU loss for positive samples
        if obj_mask.any():
            pred_pos = pred_boxes[obj_mask]  # [N_pos, 4]
            target_pos = t_box[obj_mask]     # [N_pos, 4]

            # Ensure we have valid boxes
            if len(pred_pos) > 0 and len(target_pos) > 0:
                ciou_loss = bbox_ciou(pred_pos, target_pos)
                bbox_loss = lambda_coord * ciou_loss.mean()

                # Object confidence loss
                obj_loss = lambda_obj * F.binary_cross_entropy(
                    pred_conf[obj_mask],
                    t_conf[obj_mask],
                    reduction='mean'
                )

        # Distance-weighted no-object loss
        noobj_loss = torch.tensor(0.0, device=device)
        if noobj_mask.any() and len(true_centers) > 0:
            # Create grid centers
            gx_centers = torch.arange(GX, device=device, dtype=torch.float32) + 0.5
            gy_centers = torch.arange(GY, device=device, dtype=torch.float32) + 0.5

            # Calculate minimum distance to any true center
            distances = torch.full((GY, GX), float('inf'), device=device)
            for cx, cy in true_centers:
                gx_grid, gy_grid = torch.meshgrid(gx_centers, gy_centers, indexing='xy')
                d = torch.sqrt((gx_grid - cx) ** 2 + (gy_grid.T - cy) ** 2)
                distances = torch.minimum(distances, d)

            # Distance-based weighting (farther = higher weight)
            distance_weights = torch.exp(-distances / 2.0).clamp(0.01, 1.0)
            distance_weights = distance_weights.unsqueeze(0).expand(A, -1, -1)

            # Apply weighted BCE loss
            noobj_conf = pred_conf[noobj_mask]
            noobj_targets = t_conf[noobj_mask]
            noobj_weights = distance_weights[noobj_mask]

            if len(noobj_conf) > 0:
                noobj_bce = F.binary_cross_entropy(noobj_conf, noobj_targets, reduction='none')
                noobj_loss = (noobj_bce * noobj_weights).mean() * lambda_noobj
        elif noobj_mask.any():
            # Standard no-object loss when no ground truth boxes
            noobj_loss = lambda_noobj * F.binary_cross_entropy(
                pred_conf[noobj_mask],
                t_conf[noobj_mask],
                reduction='mean'
            )

        # Accumulate losses
        batch_total_loss = bbox_loss + obj_loss + noobj_loss

        # Check for NaN/Inf values
        if torch.isnan(batch_total_loss) or torch.isinf(batch_total_loss):
            print(f"Warning: Invalid loss detected in batch {b}")
            print(f"  bbox_loss: {bbox_loss.item()}")
            print(f"  obj_loss: {obj_loss.item()}")
            print(f"  noobj_loss: {noobj_loss.item()}")
            continue

        total_loss += batch_total_loss
        total_bbox_loss += bbox_loss.item()
        total_obj_loss += obj_loss.item()
        total_noobj_loss += noobj_loss.item()

    # Average over batch
    if B > 0:
        avg_loss = total_loss / B
        avg_bbox_loss = total_bbox_loss / B
        avg_obj_loss = total_obj_loss / B
        avg_noobj_loss = total_noobj_loss / B
    else:
        avg_loss = torch.tensor(0.0, device=device)
        avg_bbox_loss = 0.0
        avg_obj_loss = 0.0
        avg_noobj_loss = 0.0

    return avg_loss, avg_obj_loss, avg_noobj_loss, avg_bbox_loss

def calculate_iou(box1, box2):
    # Inputs are [cx, cy, w, h]
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    return inter_area / (area1 + area2 - inter_area + 1e-8)


# === TRAINING FUNCTIONS ===
def train_one_epoch(model, dataloader, optimizer, device, anchors, epoch):
    model.train()
    total_loss = 0
    total_obj_loss = 0
    total_noobj_loss = 0
    total_bbox_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        
        optimizer.zero_grad()
        preds = model(imgs)
        
        # Calculate grid size from predictions
        grid_size = preds.shape[2]  # Assuming square grid
        
        loss, obj_loss, noobj_loss, bbox_loss = improved_yolo_loss(preds, targets, anchors, grid_size)
        
        if torch.isnan(loss):
            print("❌ NaN loss detected, skipping batch")
            continue
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_obj_loss += obj_loss.item() if isinstance(obj_loss, torch.Tensor) else obj_loss
        total_noobj_loss += noobj_loss.item() if isinstance(noobj_loss, torch.Tensor) else noobj_loss
        total_bbox_loss += bbox_loss.item() if isinstance(bbox_loss, torch.Tensor) else bbox_loss
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'obj': f'{obj_loss.item() if isinstance(obj_loss, torch.Tensor) else obj_loss:.4f}',
            'bbox': f'{bbox_loss.item() if isinstance(bbox_loss, torch.Tensor) else bbox_loss:.4f}'
        })
        
        if batch_idx == 0:
            decoded_preds = [decode_predictions(preds[i].unsqueeze(0), anchors) for i in range(len(imgs))]
            visualize_predictions(imgs.cpu(), decoded_preds, targets, epoch=epoch)

        
    return (total_loss / len(dataloader), 
            total_obj_loss / len(dataloader),
            total_noobj_loss / len(dataloader), 
            total_bbox_loss / len(dataloader))

@torch.no_grad()
def validate_one_epoch(model, dataloader, device, anchors):
    model.eval()
    total_loss = 0
    
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        preds = model(imgs)
        
        grid_size = preds.shape[2]
        loss, _, _, _ = improved_yolo_loss(preds, targets, anchors, grid_size)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# === PREDICTION DECODING ===
def decode_predictions(preds, anchors, conf_thresh=0.3, iou_thresh=0.5):
    """Fixed decoding with proper grid scaling and BCE-aware processing"""
    if len(preds.shape) == 5:
        preds = preds[0]  # Remove batch dimension
    
    A, GY, GX, _ = preds.shape
    device = preds.device
    
    # Generate coordinate grids
    grid_y, grid_x = torch.meshgrid(
        torch.arange(GY, device=device, dtype=torch.float32),
        torch.arange(GX, device=device, dtype=torch.float32),
        indexing="ij"
    )
    
    # Convert anchors to grid scale
    anchors_tensor = torch.tensor(anchors, device=device, dtype=torch.float32)
    anchor_w = anchors_tensor[:, 0].view(A, 1, 1) * GX  # Scaled to grid
    anchor_h = anchors_tensor[:, 1].view(A, 1, 1) * GY
    
    # Decode predictions
    pred_x = (torch.sigmoid(preds[..., 0]) + grid_x) / GX
    pred_y = (torch.sigmoid(preds[..., 1]) + grid_y) / GY

    raw_w = torch.clamp(preds[...,2], -10, 10)
    raw_h = torch.clamp(preds[...,3], -10, 10)

    pred_w = torch.exp(preds[..., 2]) * anchor_w / GX  # Normalized to 0-1
    pred_h = torch.exp(preds[..., 3]) * anchor_h / GY
    pred_conf = torch.sigmoid(preds[..., 4])
    
    # Filter by confidence
    conf_mask = pred_conf > conf_thresh
    boxes = []
    
    if conf_mask.any():
        # Get valid predictions
        x = pred_x[conf_mask]
        y = pred_y[conf_mask]
        w = pred_w[conf_mask]
        h = pred_h[conf_mask]
        conf = pred_conf[conf_mask]
        
        # Convert to corner format for NMS
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        # Apply NMS
        keep = nms(
            torch.stack([x1, y1, x2, y2], dim=1),
            conf,
            iou_thresh
        )
        
        # Convert back to center format
        for idx in keep:
            boxes.append([
                x[idx].item(),
                y[idx].item(),
                w[idx].item(),
                h[idx].item(),
                conf[idx].item()
            ])
    
    return boxes

# === VISUALIZATION ===
def visualize_predictions(images, preds, targets, epoch=None, save_prefix="eval"):
    """Visualize predictions vs ground truth — saves images instead of showing."""
    B = min(len(images), 2)
    fig, axes = plt.subplots(1, B, figsize=(B*6, 6))
    if B == 1:
        axes = [axes]

    for i in range(B):
        img = images[i].squeeze().cpu().numpy()
        H, W = img.shape

        pred_boxes = preds[i] if i < len(preds) else []
        gt_boxes = targets[i].cpu().numpy() if i < len(targets) else []

        ax = axes[i]
        ax.imshow(img, cmap='gray')

        # Predicted boxes (red)
        for box in pred_boxes:
            if len(box) >= 4:
                x, y, w, h = box[:4]
                conf = box[4] if len(box) > 4 else 1.0
                x_px, y_px = x * W, y * H
                w_px, h_px = w * W, h * H
                x1, y1 = x_px - w_px/2, y_px - h_px/2
                rect = plt.Rectangle((x1, y1), w_px, h_px, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1-5, f'{conf:.2f}', color='red', fontsize=8)

        # Ground truth boxes (green)
        for box in gt_boxes:
            if len(box) >= 4:
                x, y, w, h = box[:4]
                x_px, y_px = x * W, y * H
                w_px, h_px = w * W, h * H
                x1, y1 = x_px - w_px/2, y_px - h_px/2
                rect = plt.Rectangle((x1, y1), w_px, h_px, edgecolor='green', facecolor='none', linewidth=2)
                ax.add_patch(rect)

        title = f"Img {i+1}" + (f" (Epoch {epoch+1})" if epoch is not None else "")
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    save_name = f"{save_prefix}_epoch{epoch+1 if epoch is not None else 'final'}.png"
    save_path = os.path.join(EVAL_SAVE_PATH, "Training_visualisations", save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"📷 Saved visualization to: {save_path}")


def analyze_confidence_thresholds(model, dataset, anchors, device, num_samples=10):
    thresholds = torch.linspace(0.0, 1.0, steps=50)
    box_counts = []

    model.eval()
    with torch.no_grad():
        for conf_thresh in thresholds:
            total_boxes = 0
            
            for i in range(min(num_samples, len(dataset))):
                img, _ = dataset[i]
                img = img.unsqueeze(0).to(device)
                preds = model(img)
                
                boxes = decode_predictions(preds, anchors, conf_thresh=conf_thresh.item())
                total_boxes += len(boxes)
            
            box_counts.append(total_boxes)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds.numpy(), box_counts, marker='o')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Number of Predicted Boxes")
    plt.title(f"Confidence Threshold vs. Number of Boxes\n(on {num_samples} samples)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_losses(train_losses, val_losses, obj_losses, noobj_losses, bbox_losses):
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Train Total Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(obj_losses, label='Obj Loss', linestyle='--')
    plt.plot(noobj_losses, label='No-Obj Loss', linestyle='--')
    plt.plot(bbox_losses, label='BBox Loss', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(EVAL_SAVE_PATH, "loss_curves.png")
    plt.savefig(save_path)
    plt.show()
def evaluate_dataset_no_iou(name, model, dataset, anchors, device, conf_thresh=CONF_THRESH):
    model.eval()
    total_preds = 0
    total_gts = 0
    total_correct = 0  # count-match only

    with torch.no_grad():
        for i in range(len(dataset)):
            img, target = dataset[i]
            img = img.unsqueeze(0).to(device)

            pred = model(img)
            pred_boxes = decode_predictions(pred, anchors, conf_thresh=conf_thresh)

            num_pred = len(pred_boxes)
            num_gt = len(target)

            total_preds += num_pred
            total_gts += num_gt
            total_correct += min(num_pred, num_gt)  # count match only

    precision = total_correct / (total_preds + 1e-6)
    recall = total_correct / (total_gts + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    eval_report = (
        f"\n📊 === {name.upper()} SET EVALUATION ===\n"
        f"Total GT boxes:     {total_gts}\n"
        f"Total Predictions:  {total_preds}\n"
        f"Matched Predictions:{total_correct}\n"
        f"Precision:          {precision:.4f}\n"
        f"Recall:             {recall:.4f}\n"
        f"F1 Score:           {f1:.4f}\n"
    )

    print(eval_report)

    save_file = os.path.join(EVAL_SAVE_PATH, f"{name}_evaluation.txt")
    with open(save_file, "w") as f:
        f.write(eval_report)


# === COLLATE FUNCTION ===
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

# === MAIN TRAINING LOOP ===
if __name__ == "__main__":
    main()