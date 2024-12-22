import cv2
import torch
import os
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Set up paths
MODEL_PATH = "C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/windows model/model_final (1).pth"  # Update this path
CONFIG_PATH = "C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
IMAGE_PATH = "C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/testfloorplan3.png"  # Update this path

# Check if CUDA is available (for GPU inference)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the configuration and set model parameters
cfg = get_cfg()
cfg.merge_from_file(CONFIG_PATH)
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70  # Adjust this threshold if needed
cfg.MODEL.DEVICE = device
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (window)
# Initialize the predictor
predictor = DefaultPredictor(cfg)

# Read the input image
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("Error: Failed to read the image. Check the image path.")
    exit()

# Perform inference
outputs = predictor(image)
instances = outputs["instances"].to("cpu")

# Print prediction details
print("Predicted Classes:", instances.pred_classes)
print("Predicted Boxes:", instances.pred_boxes)

# Visualize the predictions
v = Visualizer(image[:, :, ::-1], metadata=None, scale=1.0)  # No need to update metadata if using COCO classes
out = v.draw_instance_predictions(instances)
output_image = out.get_image()[:, :, ::-1]

# Display the image using matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(output_image)
plt.axis("off")
plt.show()

# Optionally, save the output image
output_path = "output_image.jpg"
cv2.imwrite(output_path, output_image)
print(f"Output saved to {output_path}")
