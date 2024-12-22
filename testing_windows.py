import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "testfloorplan.png"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Line detection using Hough Transform
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=5, maxLineGap=1)

# Create a copy of the image for visualization
output_image = image.copy()

# Function to check if two lines are parallel
def are_parallel(line1, line2, angle_threshold=np.deg2rad(5)):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = np.arctan2(y2 - y1, x2 - x1)
    angle2 = np.arctan2(y4 - y3, x4 - x3)
    return np.abs(angle1 - angle2) < angle_threshold

# Group lines into clusters of three parallel lines
windows = []
if lines is not None:
    lines = [line[0] for line in lines]
    visited = [False] * len(lines)

    for i, line1 in enumerate(lines):
        if visited[i]:
            continue
        cluster = [line1]
        for j, line2 in enumerate(lines):
            if i != j and not visited[j] and are_parallel(line1, line2):
                # Check if lines are close enough
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                dist = np.abs((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / np.hypot(x4 - x3, y4 - y3)
                if dist < 20:  # Adjust the distance threshold based on your image scale
                    cluster.append(line2)
                    visited[j] = True

        # Check if we found exactly three parallel lines (window pattern)
        if len(cluster) == 3:
            windows.append(cluster)

# Draw the detected windows
for cluster in windows:
    for line in cluster:
        x1, y1, x2, y2 = line
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Plot the result
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Detected Windows")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Plot the detected windows' lines on a Matplotlib graph
plt.subplot(1, 2, 2)
plt.title("Window Locations")
for cluster in windows:
    for x1, y1, x2, y2 in cluster:
        plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)

plt.gca().invert_yaxis()
plt.axis("equal")
plt.show()
