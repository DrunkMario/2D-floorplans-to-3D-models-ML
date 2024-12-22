import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_black_between(contour1, contour2, image):
    """Check if the area between two contours is black."""
    # Get bounding boxes for both contours
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    # Find the bounding region between the two contours
    x_min = min(x1, x2)
    x_max = max(x1 + w1, x2 + w2)
    y_min = min(y1, y2)
    y_max = max(y1 + h1, y2 + h2)

    # Extract the region of interest (ROI) between the two contours
    roi = image[y_min:y_max, x_min:x_max]

    # Count the number of black pixels in the ROI
    black_threshold = 50  # Adjust this threshold if needed
    black_pixels = cv2.inRange(roi, (0, 0, 0), (black_threshold, black_threshold, black_threshold))

    # Calculate the percentage of black pixels
    black_ratio = np.sum(black_pixels == 255) / float(roi.size)

    # If more than 90% of the pixels are black, assume the area between contours is black
    return black_ratio > 0.9

def extend_wall_ends(contour, extension_length=5):
    """Extend the endpoints of the contour."""
    extended_contour = []
    for i in range(len(contour)):
        # Current point
        point = contour[i][0]

        # Previous point (wrap around to the last point if at the start)
        prev_point = contour[i - 1][0]

        # Vector from the previous point to the current point
        vector = np.array(point) - np.array(prev_point)

        # Normalize and extend the point
        if np.linalg.norm(vector) > 0:  # Avoid division by zero
            normalized_vector = vector / np.linalg.norm(vector)
            extended_point = point + normalized_vector * extension_length
            extended_contour.append(extended_point)

    return np.array(extended_contour, dtype=np.int32)

# Read the image
image = cv2.imread('testfloorplan.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Dilate the image to merge parallel lines (walls)
kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size based on thickness of walls
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Optionally, erode back to original thickness
eroded = cv2.erode(dilated, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a figure for plotting
plt.figure()
plt.title("Detected Walls")

# Image dimensions
img_height, img_width = image.shape[:2]

# List to store valid contours
contours_list = []

# Loop over each contour
for contour in contours:
    # Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the contour area
    area = cv2.contourArea(contour)

    # Define thresholds for filtering out the image boundary
    min_wall_area = 100  # Minimum area to be considered a wall
    image_area = img_height * img_width

    # Conditions for walls: Area, aspect ratio, and position
    if area > min_wall_area and area < image_area * 0.9:  # Exclude large contours near image boundary
        # Calculate aspect ratio
        aspect_ratio = w / float(h)

        # Define additional parameters for wall detection
        min_wall_thickness = 5  # Minimum thickness of the wall (adjust based on the image)

        # Condition for walls: large enough area and thickness
        if (w > min_wall_thickness or h > min_wall_thickness):
            # Check if the contour is close to other contours and if the region between is black
            for other_contour in contours:
                if other_contour is not contour:
                    # Check if the two contours are close to each other
                    dist = cv2.pointPolygonTest(other_contour, (x + w // 2, y + h // 2), True)

                    # If contours are close enough, check if the region between them is black
                    if abs(dist) < 20 and is_black_between(contour, other_contour, image):
                        # Merge the contours into one wall
                        contour = np.vstack((contour, other_contour))  # Merging contours
                        break
            
            # Extend the ends of the contour
            extended_contour = extend_wall_ends(contour)

            # Store the coordinates of the valid contour
            contours_list.append(extended_contour)

            # Plot the wall as a line on the graph
            for i in range(len(extended_contour)):
                start_point = extended_contour[i]
                end_point = extended_contour[(i + 1) % len(extended_contour)]
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green')

# Set labels and show the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.grid(True)
plt.show()

# Optionally, save the contours to a file
with open('contours.txt', 'w') as f:
    for contour in contours_list:
        contour_str = ','.join([f"{point[0]},{point[1]}" for point in contour])
        f.write(contour_str + '\n')
