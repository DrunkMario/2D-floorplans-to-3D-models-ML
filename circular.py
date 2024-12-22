import cv2
import numpy as np

# Load the image
image = cv2.imread('circle_image.png', cv2.IMREAD_GRAYSCALE)

# Apply a binary threshold to highlight the ellipse
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Fit an ellipse to the largest contour found
for contour in contours:
    if len(contour) >= 5:  # Fit ellipse needs at least 5 points
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        print(f"Center: {center}, Axes: {axes}, Angle: {angle}")

        # Optionally draw the detected ellipse on the image
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)
        cv2.imshow("Detected Ellipse", output_image)
        cv2.waitKey(0)
        break  # Stop after finding the first suitable ellipse
