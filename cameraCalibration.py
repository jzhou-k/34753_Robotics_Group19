import cv2
import numpy as np

# -----------------------------
# Step 1: Capture image of chessboard
# -----------------------------
image_path = "chessboard.jpg"  # replace with your image
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Chessboard dimensions (number of inner corners per row and column)
chessboard_size = (7, 5)  # e.g., 7x5 inner corners

# Detect chessboard corners
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

if not ret:
    print("Chessboard not detected.")
    exit()

# Refine corner locations
corners = cv2.cornerSubPix(
    gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)

# -----------------------------
# Step 2: Define real-world coordinates of the corners
# -----------------------------
# Assume each square is 0.03 m (3 cm) and Z=0 on the table
square_size = 0.03
world_pts = []
for i in range(chessboard_size[1]):  # rows
    for j in range(chessboard_size[0]):  # columns
        world_pts.append([j * square_size, i * square_size])

world_pts = np.array(world_pts, dtype=np.float32)
img_pts = corners.reshape(-1, 2).astype(np.float32)

# -----------------------------
# Step 3: Compute Homography
# -----------------------------
H, status = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)
print("Homography matrix:\n", H)

# -----------------------------
# Step 4: Function to convert pixel to world
# -----------------------------
def pixel_to_world(u, v, H):
    pixel = np.array([u, v, 1], dtype=np.float32)
    world_h = H @ pixel
    X = world_h[0] / world_h[2]
    Y = world_h[1] / world_h[2]
    return X, Y

# -----------------------------
# Step 5: Example usage
# -----------------------------
u, v = 100, 150  # example pixel coordinates
X, Y = pixel_to_world(u, v, H)
print(f"Pixel ({u}, {v}) -> World coordinates: X={X:.3f} m, Y={Y:.3f} m")
