import cv2
import numpy as np

image = cv2.imread('polygon.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

polygon = contours[0]

epsilon = 0.02 * cv2.arcLength(polygon, True)
approx = cv2.approxPolyDP(polygon, epsilon, True)

vertices = [tuple(pt[0]) for pt in approx]

print(f"Polygon vertices: {vertices}")
