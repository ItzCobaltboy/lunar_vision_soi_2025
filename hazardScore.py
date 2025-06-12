# Full YOLOv8 with Edge-Aware Detection, Multi-Task Learning Simulation, Hazard Scoring, Depth Estimation, and Rover Path Suggestion

# Install necessary packages first:
# pip install ultralytics opencv-python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import heapq

# Load trained YOLOv8 model
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Edge-Aware Preprocessing
def edge_aware_preprocessing(image_path):
    # Read image
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(image_gray, threshold1=50, threshold2=150)

    # Convert edge map to 3 channels
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend edge map with original image
    blended = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

    return blended, edges

# Simple Hazard Map and Path Planning
def generate_hazard_map(image_shape, detections, edge_map):
    hazard_map = np.zeros(image_shape[:2], dtype=np.float32)

    for box in detections[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        hazard_map[y1:y2, x1:x2] += 1.0  # Increase hazard score inside detected boxes

    hazard_map = cv2.GaussianBlur(hazard_map, (25, 25), 0)
    hazard_map /= np.max(hazard_map) + 1e-6  # Normalize to [0, 1]

    # Refine hazard map with edge intensity
    hazard_map *= (edge_map / 255.0)

    return hazard_map

# Simulated Depth Estimation using Edge Map
def generate_depth_map(edge_map):
    depth_map = cv2.GaussianBlur(edge_map, (25, 25), 0)
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    return depth_map

# A* Pathfinding Algorithm
def find_safe_path(hazard_map, start, goal):
    rows, cols = hazard_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= current[0] + dx < rows and 0 <= current[1] + dy < cols
        ]

        for neighbor in neighbors:
            tentative_g = g_score[current] + 1 + hazard_map[neighbor]
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None

# CLI Inference and Visualization Function
def detect_with_cam(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    # Edge-aware preprocessing
    image_bgr, edge_map = edge_aware_preprocessing(image_path)
    if image_bgr is None:
        print(f"Error: Unable to read the image at '{image_path}'. Please ensure it is a valid image file.")
        return

    detections = model.predict(image_bgr)

    if len(detections) == 0:
        print("No detections found.")
        return

    # Generate hazard map
    hazard_map = generate_hazard_map(image_bgr.shape, detections, edge_map)

    # Generate simulated depth map
    depth_map = generate_depth_map(edge_map)

    # Define start and goal points for rover (corners of the image)
    start = (0, 0)
    goal = (image_bgr.shape[0] - 1, image_bgr.shape[1] - 1)

    path = find_safe_path(hazard_map, start, goal)

    # Draw detections and path
    result_image = detections[0].plot()

    if path:
        for y, x in path:
            cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1)
    else:
        print("No safe path found.")

    # Display results using matplotlib
    plt.figure(figsize=(24, 8))

    plt.subplot(1, 3, 1)
    plt.title('YOLOv8 Detections and Path')
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Hazard Map (Edge-Aware)')
    plt.imshow(hazard_map, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Simulated Depth Map')
    plt.imshow(depth_map, cmap='viridis')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    image_path = input("Enter the path to the lunar image: ")
    detect_with_cam(image_path)

# Note: Grad-CAM and Gradio functionality are removed due to environment restrictions (missing torch and ssl modules).
# This CLI-based version now includes edge-aware preprocessing, hazard scoring, simulated depth estimation, and a simple rover path planning algorithm to demonstrate multi-task learning simulation.
