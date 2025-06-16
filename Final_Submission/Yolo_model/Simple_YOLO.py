from ultralytics import YOLO
import torch
import os


############################ A simple script to train a YOLOv8 nano, for a custom dataset of lunar craters ############################

def main():
    # Get absolute path to the Dataset folder
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Dataset_sampled")) # Adjust this path as needed

    # Escape backslashes or use raw strings for Windows paths in YAML
    train_path = BASE_DIR.replace("\\", "/") + "/train/images"
    val_path = BASE_DIR.replace("\\", "/") + "/valid/images"


    save_path = os.path.join(BASE_DIR, "yolo_model.pt")
    # Build YAML content
    yaml_content = f"""
    train: {train_path}
    val: {val_path}

    nc: 1
    names: ['crater']
    """

    # Save data.yaml in same folder as this script (or anywhere else)
    yaml_path = os.path.join(os.path.dirname(__file__), "simple_yolo_data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ data.yaml written to: {yaml_path}")

    # Verify GPU availability
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    # Load a YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load medium-sized model (you can also try 'yolov8l.pt' or 'yolov8x.pt' for larger models)


    results = model.train(
        data='simple_yolo_data.yaml',
        imgsz=416,              # Matches dataset, no resizing
        epochs=100,
        batch=-1,                
        device=0,
        workers=3,
        patience=20,
        project='lunar_craters',
        name='nano_big_dataset',
        exist_ok=True,
        amp=True,

        optimizer='AdamW',
        lr0=2e-4,
        cos_lr=True,

        single_cls=True,
        cache=False,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.7,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        close_mosaic=10
    )
    # Display training results summary

    model.save(save_path)

    print(results)

if __name__ == "__main__":
    main()