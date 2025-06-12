import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import heapq
import streamlit as st
from PIL import Image

# Load trained YOLOv8 model
try:
    model = YOLO('best.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    exit(1)

# Edge-Aware Preprocessing
def edge_aware_preprocessing(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, threshold1=50, threshold2=150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    return blended, edges

# Simple Hazard Map and Path Planning
def generate_hazard_map(image_shape, detections, edge_map):
    hazard_map = np.zeros(image_shape[:2], dtype=np.float32)
    for box in detections[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        hazard_map[y1:y2, x1:x2] += 1.0
    hazard_map = cv2.GaussianBlur(hazard_map, (25, 25), 0)
    hazard_map /= np.max(hazard_map) + 1e-6
    hazard_map *= (edge_map / 255.0)
    return hazard_map

# Simulated Depth Estimation using Edge Map
def generate_depth_map(edge_map):
    depth_map = cv2.GaussianBlur(edge_map, (25, 25), 0)
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    return depth_map

# Simulated Explainability Map
def generate_explainability_map(image_bgr):
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    explainability_map = cv2.GaussianBlur(gray_image, (99, 99), 0)
    explainability_map = cv2.normalize(explainability_map, None, 0, 1, cv2.NORM_MINMAX)
    return explainability_map

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

# Streamlit Web Interface
def web_interface():
    st.title("Lunar Crater & Boulder Detection with Path Planning and Explainability")

    uploaded_file = st.file_uploader("Choose a lunar image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        cv2.imwrite('input_image.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        image_bgr, edge_map = edge_aware_preprocessing('input_image.jpg')
        detections = model.predict(image_bgr)

        if len(detections) == 0:
            st.warning("No detections found.")
            return

        hazard_map = generate_hazard_map(image_bgr.shape, detections, edge_map)
        depth_map = generate_depth_map(edge_map)
        explainability_map = generate_explainability_map(image_bgr)

        start = (0, 0)
        goal = (image_bgr.shape[0] - 1, image_bgr.shape[1] - 1)
        path = find_safe_path(hazard_map, start, goal)

        result_image = detections[0].plot()
        if path:
            for y, x in path:
                cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1)
        else:
            st.warning("No safe path found.")

        col1, col2 = st.columns(2)

        with col1:
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption='Detections and Path', use_column_width=True)

        with col2:
            st.image(hazard_map, caption='Hazard Map (Edge-Aware)', use_column_width=True, channels='GRAY')

        col3, col4 = st.columns(2)

        with col3:
            st.image(depth_map, caption='Simulated Depth Map', use_column_width=True, channels='GRAY')

        with col4:
            st.image(explainability_map, caption='Explainability Map', use_column_width=True, channels='GRAY')

if __name__ == "__main__":
    web_interface()

# Note: Grad-CAM functionality is simulated due to environment restrictions.
# This Streamlit web interface version includes edge-aware preprocessing, hazard scoring, simulated depth estimation, simulated explainability integration, and a simple rover path planning algorithm.
