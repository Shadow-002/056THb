import cv2
import numpy as np

# Load image
img = cv2.imread('palm_vein_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply circular mask
height, width = gray.shape
mask = np.zeros((height, width), np.uint8)
cx, cy = int(width/2), int(height/2)
r = int(min(height, width)/2)
cv2.circle(mask, (cx, cy), r, (255,255,255), -1, cv2.LINE_AA)

# Apply mask to image
masked = cv2.bitwise_and(gray, gray, mask=mask)

# Extract vein pattern from each concentric circle
num_circles = 10
vein_patterns = []
for i in range(num_circles):
    r = int(min(height, width)/num_circles * (i+1))
    circle_mask = np.zeros((height, width), np.uint8)
    cv2.circle(circle_mask, (cx, cy), r, (255,255,255), -1, cv2.LINE_AA)
    circle_pattern = cv2.bitwise_and(masked, masked, mask=circle_mask)
    vein_patterns.append(circle_pattern)

# Combine vein patterns to form final ROI
final_roi = np.zeros((height, width), np.uint8)
for pattern in vein_patterns:
    final_roi = cv2.bitwise_or(final_roi, pattern)

# Display final ROI
cv2.imshow('Final ROI', final_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
