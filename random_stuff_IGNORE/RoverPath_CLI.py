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

# ===== Hazard Map and Path Planning =====
def generate_hazard_map(image_shape, detections, edge_map=None):
    hazard_map = np.zeros(image_shape[:2], dtype=np.float32)

    for box in detections[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        hazard_map[y1:y2, x1:x2] += 1.0

    hazard_map = cv2.GaussianBlur(hazard_map, (25, 25), 0)
    hazard_map /= np.max(hazard_map) + 1e-6

    if edge_map is not None:
        hazard_map *= (edge_map / 255.0)

    return hazard_map

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

# ===== Advanced Utilities =====
def edge_aware_preprocessing(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, threshold1=50, threshold2=150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    return blended, edges

def generate_depth_map(edge_map):
    depth_map = cv2.GaussianBlur(edge_map, (25, 25), 0)
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    return depth_map

def generate_explainability_map(image_bgr):
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    explainability_map = cv2.GaussianBlur(gray_image, (99, 99), 0)
    explainability_map = cv2.normalize(explainability_map, None, 0, 1, cv2.NORM_MINMAX)
    return explainability_map

# ===== Basic CLI Mode =====
def basic_mode(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Unable to read the image at '{image_path}'.")
        return

    detections = model.predict(image_bgr)
    if len(detections) == 0:
        print("No detections found.")
        return

    hazard_map = generate_hazard_map(image_bgr.shape, detections)

    start = (0, 0)
    goal = (image_bgr.shape[0] - 1, image_bgr.shape[1] - 1)
    path = find_safe_path(hazard_map, start, goal)

    result_image = detections[0].plot()

    if path:
        for y, x in path:
            cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1)
    else:
        print("No safe path found.")

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('YOLOv8 Detections and Path')
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Hazard Map')
    plt.imshow(hazard_map, cmap='hot')
    plt.axis('off')
    plt.show()

# ===== Advanced CLI Mode =====
def advanced_mode(image_path):
    image_bgr, edge_map = edge_aware_preprocessing(image_path)
    if image_bgr is None:
        print(f"Error: Unable to read the image at '{image_path}'.")
        return

    detections = model.predict(image_bgr)
    if len(detections) == 0:
        print("No detections found.")
        return

    hazard_map = generate_hazard_map(image_bgr.shape, detections, edge_map)
    depth_map = generate_depth_map(edge_map)
    explainability_map = generate_explainability_map(image_bgr)

    start = (0, 0)
    goal = (image_bgr.shape[0] - 1, image_bgr.shape[1] - 1)
    path = find_safe_path(hazard_map, start, goal)

    result_image = detections[0].plot()
    confidences = [box.conf.item() for box in detections[0].boxes]

    if path:
        for y, x in path:
            cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1)
    else:
        print("No safe path found.")

    plt.figure(figsize=(24, 12))

    plt.subplot(2, 2, 1)
    plt.title('YOLOv8 Detections and Path')
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Hazard Map (Edge-Aware)')
    plt.imshow(hazard_map, cmap='hot')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Simulated Depth Map')
    plt.imshow(depth_map, cmap='viridis')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Simulated Explainability Map')
    plt.imshow(explainability_map, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=10, range=(0, 1), color='skyblue', edgecolor='black')
    plt.title('Prediction Confidence Histogram')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# ===== Entry Point =====
if __name__ == "__main__":
    image_path = input("Enter the path to the lunar image: ")
    print("Select Mode:\n1. Basic CLI Mode\n2. Advanced CLI Mode")
    mode = input("Enter 1 or 2: ")

    if mode == '1':
        basic_mode(image_path)
    elif mode == '2':
        advanced_mode(image_path)
    else:
        print("Invalid selection. Exiting.")
