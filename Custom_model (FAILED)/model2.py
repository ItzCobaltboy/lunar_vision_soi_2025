# === COMPLETE CRATER DETECTION SCRIPT WITH ANCHOR CLUSTERING ===
import os
import torch
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from sklearn.cluster import KMeans

# === CONFIG ===
IMG_SIZE = 416
EPOCHS = 60
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CONF_THRESH = 0.70
LAMBDA_NOOBJ = 0.5
NUM_ANCHORS = 3  # Set number of anchors here
MODEL_SAVE_PATH = "crater_model.pt"
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset_sampled_pico")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VALID_IMG_DIR = os.path.join(DATASET_DIR, "valid", "images")
VALID_LABEL_DIR = os.path.join(DATASET_DIR, "valid", "labels")

# === ANCHOR GENERATION ===

class KMeansAnchorGenerator:
    def __init__(self, dataset, num_anchors=3, img_size=416):
        """
        dataset: instance of CraterDataset
        num_anchors: how many anchors to generate
        img_size: input image size used during training
        """
        self.dataset = dataset
        self.num_anchors = num_anchors
        self.img_size = img_size

    def extract_boxes(self):
        """
        Extracts all bounding boxes (w, h) from the dataset in normalized format.
        """
        whs = []
        for _, targets in tqdm(self.dataset, desc="Extracting boxes"):
            for box in targets:
                _, _, w, h = box.tolist()
                whs.append([w * self.img_size, h * self.img_size])  # convert to pixels

        return np.array(whs)

    def run_kmeans(self):
        """
        Applies KMeans clustering on bounding box widths and heights.
        Returns anchors in normalized (0-1) format.
        """
        whs = self.extract_boxes()
        if len(whs) < self.num_anchors:
            raise ValueError(f"Not enough boxes ({len(whs)}) to find {self.num_anchors} anchors.")
        
        kmeans = KMeans(n_clusters=self.num_anchors, n_init='auto', random_state=42)
        kmeans.fit(whs)
        centers = kmeans.cluster_centers_  # pixel scale

        # Normalize to 0-1 scale (for use in training)
        anchors = centers / self.img_size
        anchors = anchors.tolist()

        print("📐 Generated Anchors (w, h):")
        for i, (w, h) in enumerate(anchors):
            print(f"Anchor {i+1}: ({w:.4f}, {h:.4f})")

        return anchors
# === DATASET ===
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

        # Data augmentation
        flip = False
        if self.augment and random.random() > 0.5:
            image = cv2.flip(image, 1)
            flip = True

        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0).float()

        # Load and augment boxes
        label_path = os.path.join(self.label_dir, os.path.splitext(img_filename)[0] + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 4:
                        x, y, w, h = parts[:4]
                        if flip: x = 1 - x  # Corrected flip
                        boxes.append([x, y, w, h])

        return image, torch.tensor(boxes).float() if boxes else torch.zeros((0, 4))

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

# === MODEL ===
class CraterDetector(nn.Module):
    def __init__(self, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.MaxPool2d(2)
        )
        self.detector = nn.Conv2d(256, 5*num_anchors, kernel_size=1)

    def forward(self, x):
        x = self.features(x)  # (B, 256, GY, GX)
        x = self.detector(x)  # (B, 5*A, GY, GX)
        return x.view(x.size(0), self.num_anchors, 5, x.size(2), x.size(3)) \
                .permute(0, 3, 4, 1, 2)  # (B, GY, GX, A, 5)


# === LOSS FUNCTION ===
# === FIXED YOLO LOSS FUNCTION ===
def yolo_loss(preds, targets, anchors, device):
    B, GY, GX, A, _ = preds.shape

    # Initialize masks and targets on correct device
    obj_mask = torch.zeros((B, GY, GX, A), device=device)
    noobj_mask = torch.ones((B, GY, GX, A), device=device)
    tx = torch.zeros((B, GY, GX, A), device=device)
    ty = torch.zeros((B, GY, GX, A), device=device)
    tw = torch.zeros((B, GY, GX, A), device=device)
    th = torch.zeros((B, GY, GX, A), device=device)

    anchors_tensor = torch.tensor(anchors, device=device)

    # Build target tensors using PyTorch ops
    for b in range(B):
        for box in targets[b]:
            if len(box) < 4: continue
            x, y, w, h = box
            
            # Calculate IoU using PyTorch operations
            anchor_w = anchors_tensor[:, 0]
            anchor_h = anchors_tensor[:, 1]
            
            inter_w = torch.minimum(w, anchor_w)
            inter_h = torch.minimum(h, anchor_h)
            inter_area = inter_w * inter_h
            anchor_area = anchor_w * anchor_h
            box_area = w * h
            
            iou = inter_area / (anchor_area + box_area - inter_area + 1e-9)
            best_anchor = torch.argmax(iou)

            gi = torch.clamp((x * GX).long(), 0, GX-1)
            gj = torch.clamp((y * GY).long(), 0, GY-1)
            
            obj_mask[b, gj, gi, best_anchor] = 1
            noobj_mask[b, gj, gi, best_anchor] = 0
            
            tx[b, gj, gi, best_anchor] = x * GX - gi.float()
            ty[b, gj, gi, best_anchor] = y * GY - gj.float()
            tw[b, gj, gi, best_anchor] = torch.log(w * GX / anchors_tensor[best_anchor, 0] + 1e-4)
            th[b, gj, gi, best_anchor] = torch.log(h * GY / anchors_tensor[best_anchor, 1] + 1e-4)

    # Calculate losses
    pred_conf = torch.sigmoid(preds[..., 0])
    obj_loss = (F.binary_cross_entropy(pred_conf[obj_mask.bool()], obj_mask[obj_mask.bool()]) +
                LAMBDA_NOOBJ * F.binary_cross_entropy(pred_conf[noobj_mask.bool()], noobj_mask[noobj_mask.bool()]))
    
    coord_loss = F.mse_loss(torch.sigmoid(preds[..., 1]), tx) + \
                 F.mse_loss(torch.sigmoid(preds[..., 2]), ty) + \
                 F.mse_loss(preds[..., 3], tw) + \
                 F.mse_loss(preds[..., 4], th)
    
    return obj_loss + 5 * coord_loss


# === DECODING & NMS ===
def decode_predictions(preds, anchors, conf_thresh=CONF_THRESH, iou_thresh=0.5):
    B, GY, GX, A, _ = preds.shape
    decoded = []
    device = preds.device
    anchors_tensor = torch.tensor(anchors, device=device)

    for b in range(B):
        batch_boxes = []
        batch_scores = []
        for a in range(A):
            conf = torch.sigmoid(preds[b, ..., a, 0])
            where = torch.where(conf > conf_thresh)
            
            for gj, gi in zip(*where):
                x = (gi + torch.sigmoid(preds[b, gj, gi, a, 1])) / GX
                y = (gj + torch.sigmoid(preds[b, gj, gi, a, 2])) / GY
                w = torch.exp(preds[b, gj, gi, a, 3]) * anchors_tensor[a,0] / GX
                h = torch.exp(preds[b, gj, gi, a, 4]) * anchors_tensor[a,1] / GY
                batch_boxes.append(torch.tensor([x, y, w, h, conf[gj, gi]]))
                batch_scores.append(conf[gj, gi])
        
        if batch_boxes:
            boxes = torch.stack(batch_boxes)
            scores = torch.stack(batch_scores)
            # NMS expects boxes in [x1, y1, x2, y2] format
            boxes_xyxy = torch.stack([
                boxes[:,0] - boxes[:,2]/2,
                boxes[:,1] - boxes[:,3]/2,
                boxes[:,0] + boxes[:,2]/2,
                boxes[:,1] + boxes[:,3]/2
            ], dim=1)
            keep = nms(boxes_xyxy, scores, iou_thresh)
            decoded.append(boxes[keep].tolist())
        else:
            decoded.append([])
    return decoded[0] if B == 1 else decoded


# === VISUALIZATION ===
def plot_sample(image, pred_boxes, ax):
    image = (image.squeeze() * 255).byte().cpu().numpy()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for box in pred_boxes:
        x, y, w, h = box.cpu().numpy()
        x1 = int((x - w/2) * IMG_SIZE)
        y1 = int((y - h/2) * IMG_SIZE)
        x2 = int((x + w/2) * IMG_SIZE)
        y2 = int((y + h/2) * IMG_SIZE)
        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (0,255,0), 1)
    
    ax.imshow(img_rgb)
    ax.axis('off')

# === TRAINING SCRIPT ===
def main():

    
    # Initialize visualization
    plt.ion()
    fig, (ax_loss, ax_img) = plt.subplots(1, 2, figsize=(14,5))
    train_losses, val_losses = [], []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data setup
    train_set = CraterDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, augment=True)
    valid_set = CraterDataset(VALID_IMG_DIR, VALID_LABEL_DIR, augment=False)
    
    train_loader = DataLoader(train_set, BATCH_SIZE, True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, BATCH_SIZE, False, collate_fn=collate_fn)
    
        # Calculate anchors first
    print("⚙️ Calculating anchors...")
    anchor_generator = KMeansAnchorGenerator(train_set, num_anchors=3, img_size=IMG_SIZE)
    anchors = anchor_generator.run_kmeans()
    anchors_tensor = torch.tensor(anchors, device=device)
    print(f"📐 Calculated anchors (w,h): {anchors}")

    # Model setup
    model = CraterDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            optimizer.zero_grad()
            preds = model(images)
            loss = yolo_loss(preds, targets, anchors_tensor, device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in valid_loader:
                images = images.to(device)
                targets = [t.to(device) for t in targets]
                
                preds = model(images)
                val_loss += yolo_loss(preds, targets, anchors, device).item()
        
        # Update plots
        train_loss /= len(train_loader)
        val_loss /= len(valid_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        ax_loss.clear()
        ax_loss.plot(train_losses, 'r-', label='Train')
        ax_loss.plot(val_losses, 'b-', label='Val')
        ax_loss.legend()
        ax_loss.set_title(f"Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        
        # Visualize sample
        sample_img, _ = next(iter(valid_loader))
        with torch.no_grad():
            pred_boxes = decode_predictions(model(sample_img.to(device)), anchors)[0]
        ax_img.clear()
        plot_sample(sample_img[0], pred_boxes, ax_img)
        
        plt.pause(0.01)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved best model with val loss {val_loss:.3f}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
