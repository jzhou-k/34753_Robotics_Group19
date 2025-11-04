import cv2
import numpy as np

# Load image
img = cv2.imread("keyboard.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Noise reduction
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Load the H-edge template
template = cv2.imread("H.png", cv2.IMREAD_GRAYSCALE)
tH, tW = template.shape

# Match template (normalized correlation)
res = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)

# Set threshold for match acceptance
threshold = 0.55
loc = np.where(res >= threshold)

# Draw detections on a display copy of original keyboard image
orig = cv2.imread("keyboard.png")  # original for overlay

for pt in zip(*loc[::-1]):
    cv2.rectangle(orig, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    cv2.putText(orig, "H", (pt[0], pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# Show results
cv2.imshow("Matches", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
