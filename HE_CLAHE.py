import cv2
import numpy as np

# Load the NIR palm vein image
img = cv2.imread('palm_vein_nir_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Histogram Equalization
equalized_img = cv2.equalizeHist(img)

# Create a Circular CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=None)

# Apply Circular CLAHE to the equalized image
combined_img = clahe.apply(equalized_img)

# Display the original and enhanced images
cv2.imshow('Original Image', img)
cv2.imshow('Enhanced Image', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
