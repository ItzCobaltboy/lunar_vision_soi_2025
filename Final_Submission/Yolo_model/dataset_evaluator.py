from ultralytics import YOLO
import os
from pathlib import Path
import torch

################# A Script to infer the YOLO model and generate labels #######################

# === CONFIG ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'yolo_weights.pt')          # Trained model path
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "images")              # Folder with test images
OUTPUT_LABELS_DIR = os.path.join(BASE_DIR, "labels")            # Folder where txts will be saved

# === SETUP ===
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)

print("CUDA available:", torch.cuda.is_available())
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# === INFER AND SAVE LABELS ===
image_files = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))         # Add more extensions if needed

if not image_files:
    print("⚠️  No .jpg images found in", TEST_IMAGES_DIR)
else:
    print("============= Evaluating Model ===========")

    for img_path in image_files:
        print(f"Evaluating {img_path.name}")
        results = model(str(img_path), device=0)

        for result in results:
            txt_filename = os.path.join(OUTPUT_LABELS_DIR, f"{img_path.stem}.txt")

            with open(txt_filename, 'w') as f:
                for box in result.boxes.data:
                    cls_id = int(box[5].item())
                    x1, y1, x2, y2 = box[0:4]

                    # Convert to YOLO format
                    xc = ((x1 + x2) / 2).item()
                    yc = ((y1 + y2) / 2).item()
                    w = (x2 - x1).item()
                    h = (y2 - y1).item()
                    img_width, img_height = result.orig_shape[1], result.orig_shape[0]

                    # Normalize
                    xc /= img_width
                    yc /= img_height
                    w /= img_width
                    h /= img_height

                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print("\n✅ Evaluation complete!")
