'''import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

def is_black_between(contour1, contour2, image):
    """Check if the area between two contours is black."""
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    x_min = min(x1, x2)
    x_max = max(x1 + w1, x2 + w2)
    y_min = min(y1, y2)
    y_max = max(y1 + h1, y2 + h2)
    roi = image[y_min:y_max, x_min:x_max]
    black_threshold = 50
    black_pixels = cv2.inRange(roi, (0, 0, 0), (black_threshold, black_threshold, black_threshold))
    black_ratio = np.sum(black_pixels == 255) / float(roi.size)
    return black_ratio > 0.9

# Read the image
image_path = 'C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/testfloorplan.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Dilate the image to merge parallel lines (walls)
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Plotting setup
plt.figure()
plt.title("Detected Walls")

img_height, img_width = image.shape[:2]
contours_list = []  # List to store wall contours
windows_list = []   # List to store window contours

# Process each contour to classify as wall or window
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    min_wall_area = 100
    image_area = img_height * img_width

    if area > min_wall_area and area < image_area * 0.9:
        aspect_ratio = w / float(h)
        min_wall_thickness = 5
        if (w > min_wall_thickness or h > min_wall_thickness):
            contours_list.append((x, y, w, h))  # Save coordinates for walls
            color = 'green'  # Mark walls in green
            label = 'Wall'
            
            # Draw each contour on the plot
            for i in range(len(contour)):
                start_point = contour[i][0]
                end_point = contour[(i + 1) % len(contour)][0]
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)
            
            # Annotate on the plot
            plt.text(x + w // 2, y + h // 2, label, color=color, fontsize=8)

# Set labels and show the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Define function to perform inference and extract window coordinates
def detect_windows(image_path):
    # Set up paths for the model
    MODEL_PATH = "C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/windows model/model_final (1).pth"
    CONFIG_PATH = "C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    # Check if CUDA is available (for GPU inference)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the configuration and set model parameters
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Adjust this threshold if needed
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (window)

    # Initialize the predictor
    predictor = DefaultPredictor(cfg)

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to read the image. Check the image path.")
        exit()

    # Perform inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    

    # Print prediction details
    print("Predicted Classes:", instances.pred_classes)#
    print("Predicted Boxes:", instances.pred_boxes)

    # Visualize the predictions
    v = Visualizer(image[:, :, ::-1], metadata=None, scale=1.0)  # No need to update metadata if using COCO classes
    out = v.draw_instance_predictions(instances)
    output_image = out.get_image()[:, :, ::-1]#

    # Display the image using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()

    # Extract window coordinates
    windows = []
    for i in range(len(instances.pred_boxes)):
        box = instances.pred_boxes[i].tensor.numpy()[0]
        x1, y1, x2, y2 = box
        windows.append((x1, y1, x2 - x1, y2 - y1))

    return windows

# Detect windows using pre-trained model
windows_list = detect_windows(image_path)

# Optionally, save the contours and windows to a file
with open('C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/contours_and_windows.txt', 'w') as f:
    for contour in contours_list:
        contour_str = f"Wall: [{contour[0]},{contour[1]},{contour[2]},{contour[3]}]"
        f.write(contour_str + '\n')
    for window in windows_list:
        window_str = f"Window: [{window[0]},{window[1]},{window[2]},{window[3]}]"
        f.write(window_str + '\n')

print("Contours and window coordinates have been saved.")

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# Helper function for checking black regions
def is_black_between(contour1, contour2, image):
    """Check if the area between two contours is black."""
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    x_min = min(x1, x2)
    x_max = max(x1 + w1, x2 + w2)
    y_min = min(y1, y2)
    y_max = max(y1 + h1, y2 + h2)
    roi = image[y_min:y_max, x_min:x_max]
    black_threshold = 50
    black_pixels = cv2.inRange(roi, (0, 0, 0), (black_threshold, black_threshold, black_threshold))
    black_ratio = np.sum(black_pixels == 255) / float(roi.size)
    return black_ratio > 0.9

# File paths
image_path = 'C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/testfloorplan3.png'
walls_output_file = 'C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/wall_vertices.txt'
windows_output_file = 'C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/windows_coordinates.txt'

# Read and preprocess the image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Dilate and erode to refine wall contours
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Variables to hold data
wall_polygons = []
windows_list = []

# Process contours to extract walls
plt.figure()
plt.title("Detected Walls")
img_height, img_width = image.shape[:2]

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    min_wall_area = 100
    image_area = img_height * img_width

    if area > min_wall_area and area < image_area * 0.9:
        aspect_ratio = w / float(h)
        min_wall_thickness = 20
        if (w > min_wall_thickness or h > min_wall_thickness):
            # Store polygon vertices for walls
            polygon_vertices = [(pt[0][0], pt[0][1]) for pt in contour]
            wall_polygons.append(polygon_vertices)

            # Plot wall contours
            for i in range(len(contour)):
                start_point = contour[i][0]
                end_point = contour[(i + 1) % len(contour)][0]
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green')

# Save wall polygons to file
with open(walls_output_file, 'w') as f:
    for polygon in wall_polygons:
        for vertex in polygon:
            f.write(f"{vertex[0]}, {vertex[1]}\n")
        f.write("\n")  # Separate polygons with a blank line

print(f"Walls' coordinates saved to '{walls_output_file}'")

# Function for detecting windows
def detect_windows(image_path):
    MODEL_PATH = "C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/windows model/model_final (1).pth"
    CONFIG_PATH = "C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load configuration and model
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (window)

    predictor = DefaultPredictor(cfg)

    # Perform inference
    image = cv2.imread(image_path)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # Extract window coordinates
    windows = []
    for i in range(len(instances.pred_boxes)):
        box = instances.pred_boxes[i].tensor.numpy()[0]
        x1, y1, x2, y2 = box
        windows.append((x1, y1, x2 - x1, y2 - y1))

    return windows

# Detect windows
windows_list = detect_windows(image_path)

# Save windows data to a separate file
with open(windows_output_file, 'w') as f:
    for window in windows_list:
        f.write(f"{window[0]}, {window[1]}, {window[2]}, {window[3]}\n")

print(f"Windows' coordinates saved to '{windows_output_file}'")

# Plot and finalize visualization
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()
