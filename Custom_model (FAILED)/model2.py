# === CONFIG ===
import os
import torch

IMG_SIZE = 416
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VALID_IMG_DIR = os.path.join(DATASET_DIR, "valid", "images")
VALID_LABEL_DIR = os.path.join(DATASET_DIR, "valid", "labels")

EPOCHS = 60
BATCH_SIZE = 4  # Reduced for better gradient updates
LEARNING_RATE = 1e-3  # Reduced learning rate
WEIGHT_DECAY = 1e-4  # Added weight decay for regularization
CONF_THRESH = 0.45 # Increased confidence threshold
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "improved_crater_model.pt")

# === DATASET ===
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from torchvision.ops import nms

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
def improved_yolo_loss(preds, targets, anchors, grid_size):
    device = preds.device
    B, A, GY, GX, _ = preds.shape
    
    # Loss weights
    lambda_coord = 8.0
    lambda_noobj = 4.0
    lambda_obj = 1.0
    
    total_loss = 0.0
    total_obj_loss = 0.0
    total_noobj_loss = 0.0
    total_bbox_loss = 0.0
    
    for b in range(B):
        target = targets[b].to(device)
        pred = preds[b]  # [A, GY, GX, 5]
        
        # Initialize target tensors
        obj_mask = torch.zeros((A, GY, GX), dtype=torch.bool, device=device)
        noobj_mask = torch.ones((A, GY, GX), dtype=torch.bool, device=device)
        
        tx = torch.zeros((A, GY, GX), device=device)
        ty = torch.zeros((A, GY, GX), device=device)
        tw = torch.zeros((A, GY, GX), device=device)
        th = torch.zeros((A, GY, GX), device=device)
        tconf = torch.zeros((A, GY, GX), device=device)
        
        # Process each ground truth box
        for box in target:
            if len(box) < 4:
                continue
                
            gx, gy, gw, gh = box[:4]
            
            # Convert to grid coordinates
            gi = int(gx * GX)
            gj = int(gy * GY)
            
            # Clamp to valid grid range
            gi = max(0, min(GX - 1, gi))
            gj = max(0, min(GY - 1, gj))
            
            # Find best anchor
            best_iou = 0
            best_anchor = 0
            
            for a in range(A):
                if a >= len(anchors):
                    continue
                anchor_w, anchor_h = anchors[a]
                
                # Calculate IoU between gt box and anchor
                iou = min(gw / anchor_w, anchor_w / gw) * min(gh / anchor_h, anchor_h / gh)
                
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = a
            
            # Assign target to best anchor
            obj_mask[best_anchor, gj, gi] = True
            noobj_mask[best_anchor, gj, gi] = False
            
            # Calculate target values
            tx[best_anchor, gj, gi] = gx * GX - gi
            ty[best_anchor, gj, gi] = gy * GY - gj
            
            # Prevent log(0) by adding small epsilon
            anchor_w, anchor_h = anchors[best_anchor]
            tw[best_anchor, gj, gi] = torch.log(gw / anchor_w + 1e-8)
            th[best_anchor, gj, gi] = torch.log(gh / anchor_h + 1e-8)
            tconf[best_anchor, gj, gi] = 1.0
        
        # Extract predictions
        pred_x = torch.sigmoid(pred[..., 0])
        pred_y = torch.sigmoid(pred[..., 1])
        pred_w = pred[..., 2]
        pred_h = pred[..., 3]
        pred_conf = torch.sigmoid(pred[..., 4])
        
        # Calculate losses
        if obj_mask.any():
            # Coordinate losses (only for cells with objects)
            loss_x = F.mse_loss(pred_x[obj_mask], tx[obj_mask])
            loss_y = F.mse_loss(pred_y[obj_mask], ty[obj_mask])
            loss_w = F.mse_loss(pred_w[obj_mask], tw[obj_mask])
            loss_h = F.mse_loss(pred_h[obj_mask], th[obj_mask])
            bbox_loss = lambda_coord * (loss_x + loss_y + loss_w + loss_h)
            
            # Object confidence loss
            obj_loss = lambda_obj * F.binary_cross_entropy(pred_conf[obj_mask], tconf[obj_mask])
        else:
            bbox_loss = 0
            obj_loss = 0
        
        # No-object confidence loss
        if noobj_mask.any():
            noobj_loss = lambda_noobj * F.binary_cross_entropy(pred_conf[noobj_mask], tconf[noobj_mask])
        else:
            noobj_loss = 0
        
        batch_loss = bbox_loss + obj_loss + noobj_loss
        total_loss += batch_loss
        total_obj_loss += obj_loss if isinstance(obj_loss, torch.Tensor) else 0
        total_noobj_loss += noobj_loss if isinstance(noobj_loss, torch.Tensor) else 0
        total_bbox_loss += bbox_loss if isinstance(bbox_loss, torch.Tensor) else 0
    
    avg_loss = total_loss / B
    return avg_loss, total_obj_loss / B, total_noobj_loss / B, total_bbox_loss / B

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
        
        # Visualize occasionally (not every batch)
        # if batch_idx % max(1, len(dataloader) // 4) == 0:
        #     with torch.no_grad():
        #         decoded_preds = []
        #         for i in range(min(2, len(preds))):  # Only visualize first 2 images
        #             pred_boxes = decode_predictions(preds[i:i+1], anchors, conf_thresh=CONF_THRESH)
        #             decoded_preds.append(pred_boxes)
        #         visualize_predictions(imgs[:2], decoded_preds, targets[:2], epoch=epoch)
    
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
    """
    Decode YOLO predictions to bounding boxes
    preds: tensor of shape [1, A, GY, GX, 5] or [A, GY, GX, 5]
    """
    if len(preds.shape) == 5:
        preds = preds[0]  # Remove batch dimension
    
    A, GY, GX, _ = preds.shape
    device = preds.device
    
    # Extract confidence scores
    conf_scores = torch.sigmoid(preds[..., 4])
    
    # Filter by confidence threshold
    conf_mask = conf_scores > conf_thresh
    
    if not conf_mask.any():
        return []
    
    # Create coordinate grids
    grid_y, grid_x = torch.meshgrid(
        torch.arange(GY, device=device, dtype=torch.float32),
        torch.arange(GX, device=device, dtype=torch.float32),
        indexing="ij"
    )
    
    # Expand grids for all anchors
    grid_x = grid_x.unsqueeze(0).expand(A, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(A, -1, -1)
    
    # Convert anchors to tensor
    anchors_tensor = torch.tensor(anchors, device=device, dtype=torch.float32)
    anchor_w = anchors_tensor[:, 0].view(A, 1, 1)
    anchor_h = anchors_tensor[:, 1].view(A, 1, 1)
    
    # Decode predictions
    pred_x = (torch.sigmoid(preds[..., 0]) + grid_x) / GX
    pred_y = (torch.sigmoid(preds[..., 1]) + grid_y) / GY
    pred_w = torch.exp(preds[..., 2]) * anchor_w
    pred_h = torch.exp(preds[..., 3]) * anchor_h
    pred_conf = conf_scores
    
    # Apply confidence mask
    boxes_x = pred_x[conf_mask]
    boxes_y = pred_y[conf_mask]
    boxes_w = pred_w[conf_mask]
    boxes_h = pred_h[conf_mask]
    boxes_conf = pred_conf[conf_mask]
    
    if len(boxes_x) == 0:
        return []
    
    # Convert to corner format for NMS
    x1 = boxes_x - boxes_w / 2
    y1 = boxes_y - boxes_h / 2
    x2 = boxes_x + boxes_w / 2
    y2 = boxes_y + boxes_h / 2
    
    boxes_corner = torch.stack([x1, y1, x2, y2], dim=1)
    
    # Apply NMS
    keep = nms(boxes_corner, boxes_conf, iou_thresh)
    
    # Convert back to center format
    final_boxes = []
    for idx in keep:
        final_boxes.append([
            boxes_x[idx].item(),
            boxes_y[idx].item(), 
            boxes_w[idx].item(),
            boxes_h[idx].item(),
            boxes_conf[idx].item()
        ])
    
    return final_boxes

# === VISUALIZATION ===
def visualize_predictions(images, preds, targets, epoch=None):
    """Visualize predictions vs ground truth"""
    B = min(len(images), 2)  # Limit to 2 images
    
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
        
        # Plot predicted boxes (RED)
        for box in pred_boxes:
            if len(box) >= 4:
                x, y, w, h = box[:4]
                conf = box[4] if len(box) > 4 else 1.0
                
                # Convert to pixel coordinates
                x_px, y_px = x * W, y * H
                w_px, h_px = w * W, h * H
                x1, y1 = x_px - w_px/2, y_px - h_px/2
                
                rect = plt.Rectangle((x1, y1), w_px, h_px, 
                                   edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1-5, f'{conf:.2f}', color='red', fontsize=8)
        
        # Plot ground truth boxes (GREEN)
        for box in gt_boxes:
            if len(box) >= 4:
                x, y, w, h = box[:4]
                
                # Convert to pixel coordinates
                x_px, y_px = x * W, y * H
                w_px, h_px = w * W, h * H
                x1, y1 = x_px - w_px/2, y_px - h_px/2
                
                rect = plt.Rectangle((x1, y1), w_px, h_px,
                                   edgecolor='green', facecolor='none', linewidth=2)
                ax.add_patch(rect)
        
        title = f"Img {i+1}" + (f" (Epoch {epoch+1})" if epoch is not None else "")
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(500)
    # plt.close()

# === COLLATE FUNCTION ===
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

# === MAIN TRAINING LOOP ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets with augmentation for training
    train_dataset = CraterDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=True)
    valid_dataset = CraterDataset(VALID_IMG_DIR, VALID_LABEL_DIR, augment=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # Generate anchors using K-means
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
    
    # for epoch in range(EPOCHS):
    #     print(f"\n--- EPOCH {epoch+1}/{EPOCHS} ---")
        
    #     # Training
    #     train_loss, obj_loss, noobj_loss, bbox_loss = train_one_epoch(
    #         model, train_loader, optimizer, device, ANCHORS, epoch)
        
    #     # Validation
    #     val_loss = validate_one_epoch(model, valid_loader, device, ANCHORS)
        
    #     # Learning rate scheduling
    #     scheduler.step(val_loss)
        
    #     print(f"Train Loss: {train_loss:.4f} (obj: {obj_loss:.4f}, noobj: {noobj_loss:.4f}, bbox: {bbox_loss:.4f})")
    #     print(f"Val Loss: {val_loss:.4f}")
    #     print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
    #     # Save best model
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save({
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'epoch': epoch,
    #             'val_loss': val_loss,
    #             'anchors': ANCHORS
    #         }, MODEL_SAVE_PATH)
    #         print("✅ Best model saved!")
    #         patience = 0
    #     else:
    #         patience += 1
            
    #     # Early stopping
    #     if patience >= max_patience:
    #         print(f"Early stopping after {max_patience} epochs without improvement")
    #         break
    
    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test on a few samples
    with torch.no_grad():
        for i in range(min(3, len(train_dataset))):
            sample_img, sample_target = train_dataset[i]
            preds = model(sample_img.unsqueeze(0).to(device))
            boxes = decode_predictions(preds, ANCHORS, conf_thresh=CONF_THRESH)
            
            print(f"Sample {i+1}: Found {len(boxes)} boxes")
            visualize_predictions([sample_img], [boxes], [sample_target])
    
    print("Training completed!")