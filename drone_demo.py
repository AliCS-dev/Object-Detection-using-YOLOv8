import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import defaultdict
from itertools import combinations

# --- Load YOLOv8x Model ---
model = YOLO("yolov8x.pt")

# --- Parameters ---~
image_path = "Pic1.jpg"           # Input drone image
tile_size = 1280                  # Each tile size for processing
overlap = 0.4                     # 40% overlap between tiles
conf_threshold = 0.25             # Minimum confidence for detection
neighbors_k = 2                   # Draw lines to K nearest neighbors per object

# --- Load the Original Image ---
original_img = cv2.imread(image_path)


height, width, _ = original_img.shape

# Compute stride from tile size and overlap
stride = int(tile_size * (1 - overlap))

# Initialize data containers
all_detections = []              # Store detections across all tiles
class_counts = defaultdict(int)  # Count objects by class
heatmap_img = np.zeros((height, width), dtype=np.float32)  # For heatmap

# --- TILE-BASED DETECTION LOOP ---
for y in range(0, height, stride):
   
    for x in range(0, width, stride):
        x_end = min(x + tile_size, width)
        y_end = min(y + tile_size, height)
        tile = original_img[y:y_end, x:x_end]

        # Run YOLO on this tile
        results = model(tile, imgsz=tile_size, conf=conf_threshold, augment=True)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])  # Class ID
            class_name = model.names[cls_id]  # Human-readable name
            class_counts[class_name] += 1

            # Local box coords (in tile)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Adjust box to global image coords
            global_x1 = x + x1
            global_y1 = y + y1
            global_x2 = x + x2
            global_y2 = y + y2
            global_cx = (global_x1 + global_x2) // 2
            global_cy = (global_y1 + global_y2) // 2

            # Store detection
            all_detections.append({
                "box": (global_x1, global_y1, global_x2, global_y2),
                "label": class_name,
                "center": (global_cx, global_cy)
            })

            # Update heatmap intensity at center point
            if 0 <= global_cy < height and 0 <= global_cx < width:
                heatmap_img[global_cy, global_cx] += 1

# --- Prepare Copies for Drawing ---
output_img = original_img.copy()

# --- DRAW BOUNDING BOXES, LABELS, CENTERS ---
for det in all_detections:
    x1, y1, x2, y2 = det["box"]
    label = det["label"]
    cx, cy = det["center"]
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
    cv2.circle(output_img, (cx, cy), 3, (0, 255, 255), -1)         # Yellow center dot
    cv2.putText(output_img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)     # Class label

# --- FUNCTION: DRAW CONNECTIONS TO K NEAREST OBJECTS ---
def draw_connections(img, detections):
    centers = [d["center"] for d in detections]
    for i, center_a in enumerate(centers):
        distances = [(j, math.hypot(center_b[0] - center_a[0], center_b[1] - center_a[1]))
                     for j, center_b in enumerate(centers) if i != j]
        nearest = sorted(distances, key=lambda x: x[1])[:neighbors_k]
        for j, dist in nearest:
            cx1, cy1 = center_a
            cx2, cy2 = centers[j]
            mid_x, mid_y = (cx1 + cx2) // 2, (cy1 + cy2) // 2
            cv2.line(img, (cx1, cy1), (cx2, cy2), (255, 255, 0), 1)  # Cyan line
            cv2.putText(img, f"{int(dist)}px", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

draw_connections(output_img, all_detections)

# --- DRAW SUMMARY TEXT ---
summary = f"Total: {sum(class_counts.values())} | " + ", ".join(
    [f"{cls}: {count}" for cls, count in class_counts.items()]
)
cv2.putText(output_img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

# --- ADD GRID TO OUTPUT ---
grid_rows, grid_cols = 6, 4
cell_w = width // grid_cols
cell_h = height // grid_rows
grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

# Draw grid lines
for i in range(1, grid_rows):
    cv2.line(output_img, (0, i * cell_h), (width, i * cell_h), (255, 255, 255), 1)
for j in range(1, grid_cols):
    cv2.line(output_img, (j * cell_w, 0), (j * cell_w, height), (255, 255, 255), 1)

# Count and label objects in each grid cell
for det in all_detections:
    cx, cy = det["center"]
    row = min(cy // cell_h, grid_rows - 1)
    col = min(cx // cell_w, grid_cols - 1)
    grid_counts[row, col] += 1

for i in range(grid_rows):
    for j in range(grid_cols):
        cv2.putText(output_img, str(grid_counts[i, j]), (j * cell_w + 5, i * cell_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# --- SAVE BOUNDING BOX OUTPUT ---
cv2.imwrite("Output1.jpg", output_img)

# --- HEATMAP CREATION ---
# Step 1: Blur the intensity points to create density
heatmap_blur = cv2.GaussianBlur(heatmap_img, (0, 0), sigmaX=25, sigmaY=25)

# Step 2: Normalize to 0-255 and colorize
heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

# Step 3: Blend with original image
heatmap_overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# Step 4: Add grid and connections on heatmap too
for i in range(1, grid_rows):
    cv2.line(heatmap_overlay, (0, i * cell_h), (width, i * cell_h), (255, 255, 255), 1)
for j in range(1, grid_cols):
    cv2.line(heatmap_overlay, (j * cell_w, 0), (j * cell_w, height), (255, 255, 255), 1)

draw_connections(heatmap_overlay, all_detections)

# --- SAVE HEATMAP OUTPUT ---
cv2.imwrite("Heat1.jpg", heatmap_overlay)

# --- SHOW RESULTS ---
cv2.imshow("Detections", output_img)
cv2.imshow("Heatmap", heatmap_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- PRINT SUMMARY TO CONSOLE ---
print("\n📊 Detection Summary:")
print(f"Total objects detected: {sum(class_counts.values())}")
for cls, count in class_counts.items():
    print(f"- {cls}: {count}")