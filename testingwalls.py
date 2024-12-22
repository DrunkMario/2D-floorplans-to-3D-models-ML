'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
vertex_list = []  # List to store coordinates of vertices

# Process each contour to classify as wall
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
            
            # Loop through the contour points to plot and extract the coordinates
            for i in range(len(contour)):
                start_point = contour[i][0]
                end_point = contour[(i + 1) % len(contour)][0]
                
                # Plot the contour
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)
                
                # Store the vertex coordinates in the vertex_list
                vertex_list.append((start_point[0], start_point[1]))

            # Annotate the wall on the plot
            plt.text(x + w // 2, y + h // 2, label, color=color, fontsize=8)

# Save coordinates to a file
output_file = 'C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/wall_vertices.txt'
with open(output_file, 'w') as f:
    for vertex in vertex_list:
        f.write(f'{vertex[0]}, {vertex[1]}\n')

# Set labels and show the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

print(f"Coordinates of the vertices are saved to '{output_file}'")
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
polygon_list = []  # List to store polygons, each containing its vertices

# Process each contour to classify as wall
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
            
            # Store the current polygon's vertices
            polygon_vertices = []
            
            # Loop through the contour points to plot and extract the coordinates
            for i in range(len(contour)):
                start_point = contour[i][0]
                end_point = contour[(i + 1) % len(contour)][0]
                
                # Plot the contour
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)
                
                # Add the vertex to the current polygon
                polygon_vertices.append((start_point[0], start_point[1]))

            # Append the polygon to the list of polygons
            polygon_list.append(polygon_vertices)

            # Annotate the wall on the plot
            plt.text(x + w // 2, y + h // 2, label, color=color, fontsize=8)

# Save coordinates to a file
output_file = 'C:/Users/Aryan/OneDrive/Desktop/My Codes/floor planner/wall_vertices.txt'
with open(output_file, 'w') as f:
    for polygon in polygon_list:
        for vertex in polygon:
            f.write(f'{vertex[0]}, {vertex[1]}\n')
        f.write('\n')  # Separate each polygon with a blank line

# Set labels and show the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

print(f"Coordinates of the polygons' vertices are saved to '{output_file}'")
