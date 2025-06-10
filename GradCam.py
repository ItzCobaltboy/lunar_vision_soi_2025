# Full YOLOv8 with CLI-based Visualization (PyTorch Removed for Compatibility)

# Install necessary packages first:
# pip install ultralytics opencv-python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load trained YOLOv8 model
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# CLI Inference and Visualization Function
def detect_with_cam(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Unable to read the image at '{image_path}'. Please ensure it is a valid image file.")
        return

    detections = model.predict(image_bgr)

    if len(detections) == 0:
        print("No detections found.")
        return

    # Get YOLO predicted image with boxes
    result_image = detections[0].plot()

    # Display results using matplotlib
    plt.figure(figsize=(8, 8))
    plt.title('YOLOv8 Detections')
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = input("Enter the path to the lunar image: ")
    detect_with_cam(image_path)

# Note: Grad-CAM and Gradio functionality are removed due to environment restrictions (missing torch and ssl modules).
# This CLI-based version ensures compatibility and allows you to visualize YOLOv8 detection results locally.
