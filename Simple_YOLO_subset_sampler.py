import os
import random
import shutil

def sample_dataset(split, sample_size):
    current_dir = os.path.dirname(__file__)
    
    src_img_dir = os.path.join(current_dir, "Dataset", split, "images")
    src_label_dir = os.path.join(current_dir, "Dataset", split, "labels")
    
    dst_img_dir = os.path.join(current_dir, "Dataset_sampled", split, "images")
    dst_label_dir = os.path.join(current_dir, "Dataset_sampled", split, "labels")
    
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(src_img_dir) if f.endswith((".jpg", ".png"))]
    sampled_imgs = random.sample(all_imgs, min(sample_size, len(all_imgs)))

    for img_file in sampled_imgs:
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"

        src_img_path = os.path.join(src_img_dir, img_file)
        src_label_path = os.path.join(src_label_dir, label_file)

        dst_img_path = os.path.join(dst_img_dir, img_file)
        dst_label_path = os.path.join(dst_label_dir, label_file)

        shutil.copyfile(src_img_path, dst_img_path)
        if os.path.exists(src_label_path):
            shutil.copyfile(src_label_path, dst_label_path)

    print(f"✅ Sampled {len(sampled_imgs)} from '{split}' to '{dst_img_dir}'")

# === CONFIG ===
sample_dataset("train", sample_size=300)
sample_dataset("valid", sample_size=200)
