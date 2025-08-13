# Drone Object Detection & Heatmap Visualization Using YOLOv8

This project performs high-accuracy object detection on drone images using the **YOLOv8x** model. It processes large images in tiles, draws bounding boxes, computes K-nearest neighbor connections, overlays a grid, and generates a heatmap showing object density.

---

## 🔹 Features

- **Tile-based detection** for large images to handle high-resolution drone imagery.
- **YOLOv8x object detection** with confidence thresholding.
- **Bounding boxes** with object labels and center points.
- **K-nearest neighbor connections** between detected objects.
- **Grid-based counting** of objects per section of the image.
- **Heatmap visualization** showing object density across the image.
- **Summary statistics** printed to console and overlaid on images.
- **Automatic saving** of detection and heatmap outputs.

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO (`ultralytics`)
- YOLOv8x model weights (`yolov8x.pt`)

Install dependencies via pip:

```bash
pip install opencv-python numpy ultralytics
Ensure you have the YOLOv8x model file (yolov8x.pt) in your project directory.

📂 Project Structure
bash
Copy
Edit
project/
│
├─ Pic1.jpg                  # Input drone image
├─ yolov8x.pt                # YOLOv8x model
├─ main.py                   # Detection and visualization script
├─ Output1.jpg               # Output image with bounding boxes
├─ Heat1.jpg                 # Heatmap visualization
└─ README.md                 # Project documentation
⚡ Usage
Place your input image in the project directory.

Adjust parameters in the script:

python
Copy
Edit
image_path = "Pic1.jpg"        # Input image path
tile_size = 1280               # Tile size for processing
overlap = 0.4                  # Overlap between tiles (0-1)
conf_threshold = 0.25          # Minimum detection confidence
neighbors_k = 2                # Number of nearest neighbors to connect
Run the script:

bash
Copy
Edit
python main.py
Outputs:

Output1.jpg → Original image with bounding boxes, labels, centers, K-nearest neighbor lines, and grid counts.

Heat1.jpg → Heatmap overlay with object density, grid, and connections.

Console prints detection summary.

🔹 How It Works
Tile-based detection:

Large images are split into overlapping tiles.

YOLOv8x is run on each tile to detect objects.

Detected boxes are converted from local tile coordinates to global image coordinates.

Bounding boxes & labels:

Each detected object is drawn with a red bounding box, yellow center dot, and label.

K-nearest neighbor connections:

Lines are drawn connecting each object to its K nearest neighbors.

Distance in pixels is displayed on each connection.

Grid-based counting:

Image is divided into a configurable grid (6x4 by default).

Each cell shows the count of objects in that region.

Heatmap visualization:

Centers of detected objects contribute to a heatmap.

Gaussian blur is applied for smooth density visualization.

Heatmap is blended with the original image.

Summary:

Prints total objects detected and count per class.

🔹 Example Output
Bounding Box Image (Output1.jpg):



Heatmap Image (Heat1.jpg):



⚙️ Parameters You Can Adjust
Parameter	Description	Example
tile_size	Size of image tiles for YOLO processing	1280
overlap	Overlap ratio between tiles (0–1)	0.4
conf_threshold	Minimum confidence for object detection	0.25
neighbors_k	Number of nearest neighbors to connect	2
grid_rows, grid_cols	Grid size for regional object counting	6, 4

⚡ Notes
Large model files (yolov8x.pt) should be handled using Git LFS if storing in a GitHub repository.

Adjust tile_size and overlap based on your image resolution and GPU memory.

The heatmap can help quickly identify areas with high object density.

🔹 References
Ultralytics YOLOv8 Documentation

OpenCV Python Documentation

👨‍💻 Author
AliCS-dev – Summer School 2025 Project
```
