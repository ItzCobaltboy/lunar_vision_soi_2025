# === CONFIG ===
import os
import torch

IMG_SIZE = 416
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset_sampled_pico")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VALID_IMG_DIR = os.path.join(DATASET_DIR, "valid", "images")
VALID_LABEL_DIR = os.path.join(DATASET_DIR, "valid", "labels")

EPOCHS = 100
BATCH_SIZE = 5
LEARNING_RATE = 1e-2
CONF_THRESH = 0.15
MODEL_SAVE_PATH = os.path.join(BASE_DIR ,"cutom_crater_model.pt")



trainCOUNT = 0 


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
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
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
                    if len(parts) == 5:
                        _, x, y, w, h = parts
                        boxes.append([x, y, w, h])

        boxes = torch.tensor(boxes).float() if boxes else torch.zeros((0, 4))
        return image, boxes
    


# === MODEL ===

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



class CraterDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.detect = nn.Conv2d(64, 15, 1)  # 3 anchors * (x, y, w, h, conf)

    def forward(self, x):
        x = self.features(x)
        x = self.detect(x)  # [B, 15, 52, 52]
        B = x.shape[0]
        x = x.view(B, 3, 5, 52, 52).permute(0, 1, 3, 4, 2)  # [B, 3, 52, 52, 5]
        return x

# === IOU and LOSS ===
def bbox_iou(box1, box2):
    """
    IoU for [cx, cy, w, h] format boxes
    """
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0


def yolo_loss(preds, targets, anchors, grid_size=52):
    device = preds.device
    B, A, GY, GX, _ = preds.shape
    loss = 0.0
    lambda_coord = 5.0
    lambda_noobj = 1.0

    for b in range(B):
        target = targets[b].to(device)
        pred = preds[b]  # [A, GY, GX, 5]

        obj_mask = torch.zeros((A, GY, GX), dtype=torch.bool, device=device)
        txs = torch.zeros((A, GY, GX), device=device)
        tys = torch.zeros((A, GY, GX), device=device)
        tws = torch.zeros((A, GY, GX), device=device)
        ths = torch.zeros((A, GY, GX), device=device)
        tconfs = torch.zeros((A, GY, GX), device=device)

        for box in target:
            gx, gy, gw, gh = box
            gi = int(gx * GX)
            gj = int(gy * GY)

            best_iou = 0
            best_a = 0
            for a in range(A):
                anchor_w, anchor_h = anchors[a]
                anchor_box = torch.tensor([0, 0, anchor_w, anchor_h], device=device)
                gt_box = torch.tensor([0, 0, gw, gh], device=device)
                iou = bbox_iou(anchor_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_a = a

            # Avoid assigning the same grid cell + anchor twice
            if obj_mask[best_a, gj, gi]:
                continue

            obj_mask[best_a, gj, gi] = 1
            txs[best_a, gj, gi] = gx * GX - gi
            tys[best_a, gj, gi] = gy * GY - gj
            tws[best_a, gj, gi] = torch.log(gw / anchors[best_a][0] + 1e-6)
            ths[best_a, gj, gi] = torch.log(gh / anchors[best_a][1] + 1e-6)
            tconfs[best_a, gj, gi] = 1.0

        pred_x = torch.sigmoid(pred[..., 0])
        pred_y = torch.sigmoid(pred[..., 1])
        pred_w = pred[..., 2]
        pred_h = pred[..., 3]
        pred_conf = torch.sigmoid(pred[..., 4])

        # Only calculate loss where object is present
        # Handle potential empty tensors safely
        loss_x = F.mse_loss(pred_x[obj_mask], txs[obj_mask]) if obj_mask.any() else 0
        loss_y = F.mse_loss(pred_y[obj_mask], tys[obj_mask]) if obj_mask.any() else 0
        loss_w = F.mse_loss(pred_w[obj_mask], tws[obj_mask]) if obj_mask.any() else 0
        loss_h = F.mse_loss(pred_h[obj_mask], ths[obj_mask]) if obj_mask.any() else 0

        # BCE for objectness
        loss_obj = F.binary_cross_entropy(pred_conf[obj_mask], tconfs[obj_mask]) if obj_mask.any() else 0
        loss_noobj = lambda_noobj * F.binary_cross_entropy(pred_conf[~obj_mask], tconfs[~obj_mask]) if obj_mask.any() else 0

        # Normalize by number of positive samples
        num_obj = obj_mask.sum().float().clamp(min=1.0)  # avoid divide-by-zero
        total = (loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj) / num_obj

        loss += total

    return loss / B



# === TRAINING ===
def train_one_epoch(model, dataloader, optimizer, device, anchors):

    model.train()

    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        preds = model(imgs)

        loss = yolo_loss(preds, targets, anchors)

        if torch.isnan(loss):
            print("❌ NaN loss detected, skipping batch")
            continue  # Skip this batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        pbar.update(1)


        decoded_preds = [decode_predictions(p.unsqueeze(0), conf_thresh=CONF_THRESH) for p in preds.cpu()]    
        visualize_ground_truth_vs_prediction(imgs.cpu(), decoded_preds, targets)
    return total_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, device, anchors):
    model.eval()
    total_loss = 0
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        preds = model(imgs)
        loss = yolo_loss(preds, targets, anchors)
        total_loss += loss.item()
    return total_loss / len(dataloader)


# === COLLATE FN ===
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def show_predictions(image_tensor, boxes):
    """
    image_tensor: torch.Tensor shape [1, H, W]
    boxes: list of [cx, cy, w, h, conf]
    """
    img = image_tensor.squeeze(0).cpu().numpy()  # [H, W]
    h, w = img.shape

    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')

    for box in boxes:
        cx, cy, bw, bh, conf = box
        x = (cx - bw/2) * w
        y = (cy - bh/2) * h
        width = bw * w
        height = bh * h
        rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, f'{conf:.2f}', color='yellow', fontsize=8)

    plt.title("Predicted Crater Boxes")
    plt.show()


def decode_predictions(preds, conf_thresh=CONF_THRESH, iou_thresh=0.5):
    B, A, GY, GX, _ = preds.shape
    preds = preds[0]  # batch size 1
    device = preds.device

    conf = torch.sigmoid(preds[..., 4])
    print("avg conf:", conf.mean().item(), "| max conf:", conf.max().item(), "| min conf:", conf.min().item())
    mask = conf > conf_thresh
    if not mask.any():
        return []

    grid_y, grid_x = torch.meshgrid(
        torch.arange(GY, device=device),
        torch.arange(GX, device=device),
        indexing="ij"
    )
    grid_x = grid_x.unsqueeze(0).expand(A, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(A, -1, -1)

    anchors = torch.tensor(ANCHORS, device=device)
    anchor_w = anchors[:, 0].view(A, 1, 1)
    anchor_h = anchors[:, 1].view(A, 1, 1)

    x = (torch.sigmoid(preds[..., 0]) + grid_x) 
    y = (torch.sigmoid(preds[..., 1]) + grid_y)
    w = torch.exp(preds[..., 2]) * anchor_w 
    h = torch.exp(preds[..., 3]) * anchor_h 

    x, y, w, h, conf = x[mask], y[mask], w[mask], h[mask], conf[mask]

    boxes = torch.stack([
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ], dim=1)

    scores = conf

    keep = nms(boxes, scores, iou_thresh)
    final_boxes = torch.stack([x, y, w, h, conf], dim=1)[keep]

    return final_boxes.tolist()

def visualize_ground_truth_vs_prediction(images, preds, targets, epoch=None, pause_time=0.1):
    """
    Visualizes predicted vs ground truth boxes on grayscale images.
    Args:
        images: Tensor of shape (B, 1, H, W)
        preds: List of predicted boxes per image [ [ [x, y, w, h, conf], ... ], ... ]
        targets: List of ground truth boxes per image [ [ [x, y, w, h], ... ], ... ]
        epoch: Optional epoch number for title
        pause_time: Time to pause for non-blocking display
    """
    B = len(images)
    for i in range(1):
        img = images[i].squeeze().cpu().numpy()
        H, W = img.shape

        pred_boxes = preds[i]
        gt_boxes = targets[i]

        # Convert tensors to numpy if needed
        if torch.is_tensor(pred_boxes):
            pred_boxes = pred_boxes.cpu().numpy()
        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        # Plot predicted boxes in RED
        for box in pred_boxes:
            x, y, w, h = box[:4]
            # Denormalize
            x *= W
            y *= H
            w *= W
            h *= H
            x1, y1 = x - w / 2, y - h / 2
            rect = plt.Rectangle((x1, y1), w, h, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)

        # Plot ground truth boxes in GREEN
        for box in gt_boxes:
            x, y, w, h = box[:4]
            # Denormalize
            x *= W
            y *= H
            w *= W
            h *= H
            x1, y1 = x - w / 2, y - h / 2
            rect = plt.Rectangle((x1, y1), w, h, edgecolor='green', facecolor='none', linewidth=2)
            ax.add_patch(rect)

        title = f"Predictions vs GT - Img {i}" + (f" (Epoch {epoch})" if epoch is not None else "")
        plt.title(title)
        plt.axis("off")
        plt.pause(pause_time)
        plt.close(fig)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CraterDetector().to(device)

    train_dataset = CraterDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
    valid_dataset = CraterDataset(VALID_IMG_DIR, VALID_LABEL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=3, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=3, pin_memory=True)

    
    anchor_generator = KMeansAnchorGenerator(train_dataset, num_anchors=3, img_size=IMG_SIZE)
    ANCHORS = anchor_generator.run_kmeans()


    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"\n--- EPOCH {epoch+1}/{EPOCHS} ---")
        trainCOUNT = epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device, ANCHORS)
        val_loss = validate_one_epoch(model, valid_loader, device, ANCHORS)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Model saved!")

    # Evaluation

    print("Evaluating")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("Loaded model")
    model.eval()
    sample_img, _ = train_dataset[0]
    with torch.no_grad():
        preds = model(sample_img.unsqueeze(0).to(device))
        print("Successfully retrieved predictions from ML, decoding")
        # print(preds)

    boxes = decode_predictions(preds.cpu())
    print(f"Kept {len(boxes)} boxes above confidence threshold")

    # print("📦 Predicted boxes:", boxes)
    show_predictions(sample_img, boxes)
    print("Script Exited")