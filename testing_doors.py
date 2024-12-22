import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/testfloorplan3.png"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the image for visualization
output_image = image.copy()

def is_door_pattern(contour):
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    # Check for a rectangular shape
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # Check for aspect ratio typical of a window (e.g., long and narrow)
        if 2.0 < aspect_ratio < 10.0 and 20 < w < 200 and 5 < h < 50:
            return True
    return False

# List to store the coordinates of detected windows
window_coords = []

# Iterate through contours to detect windows
for contour in contours:
    if is_door_pattern(contour):
        x, y, w, h = cv2.boundingRect(contour)
        window_coords.append((x, y, w, h))
        # Draw the detected window on the image
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Plot the result
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Detected Doors")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Plot the windows on a Matplotlib graph
plt.subplot(1, 2, 2)
plt.title("Door Locations")
for (x, y, w, h) in window_coords:
    plt.plot([x, x + w], [y, y], color='blue', linewidth=2)
    plt.plot([x + w, x + w], [y, y + h], color='blue', linewidth=2)
    plt.plot([x + w, x], [y + h, y + h], color='blue', linewidth=2)
    plt.plot([x, x], [y + h, y], color='blue', linewidth=2)

plt.gca().invert_yaxis()
plt.axis("equal")
plt.show()
