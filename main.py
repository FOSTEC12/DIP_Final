import cv2
import numpy as np

# Load and preprocess image
image = cv2.imread(r'./images/sample1.png')
cv2.imshow('Original Image', image)  # Display the original image
cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)  # Display the grayscale conversion
cv2.waitKey(0)

# Apply Gaussian Blur
preprocessed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
cv2.imshow('Blurred Image', preprocessed_image)  # Display the blurred image
cv2.waitKey(0)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening_image = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, kernel)
closing_image = cv2.morphologyEx(opening_image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Opening + Closing Image', closing_image)  # Display after opening and closing
cv2.waitKey(0)

# Threshold to create binary image
_, thresholded_image = cv2.threshold(closing_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded Image', thresholded_image)  # Display the thresholded image
cv2.waitKey(0)

# Find contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area (consider adjusting these thresholds based on your specific needs)
min_area = 0  # Adjusted based on experiment
max_area = 500  # Adjusted based on experiment
filtered_contours = [contour for contour in contours if min_area <= cv2.contourArea(contour) <= max_area]

# Visualize filtered contours in red
filtered_contour_image = image.copy()
cv2.drawContours(filtered_contour_image, filtered_contours, -1, (0, 0, 255), 2)
cv2.imshow('Filtered Contours in Red', filtered_contour_image)  # Display the image with red contours
cv2.waitKey(0)

# Count cells based on filtered contours
filtered_cell_count = len(filtered_contours)

# Calculate coverage
dish_area = gray_image.shape[0] * gray_image.shape[1]
filtered_cell_area = sum(cv2.contourArea(contour) for contour in filtered_contours)
coverage = (filtered_cell_area / dish_area) * 100

# Display filtered cell count and coverage
print(f"Filtered Cell Count: {filtered_cell_count}")
print(f"Coverage: {coverage}%")

cv2.destroyAllWindows()  # Close all windows
