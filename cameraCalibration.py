import cv2
import numpy as np

# -----------------------------
# Step 1: Prepare your points
# -----------------------------

# Example: pixel coordinates (u,v) of keyboard keys in the camera image
# Format: Nx2 array
pixel_points = np.array([
    [281, 333],  # key H
    [200, 333],  # key G
    [362, 333],  # key J
    [240, 410],  # key B
    [342, 250],  # key U
    [34, 330],  # key D
], dtype=np.float32)

# Corresponding real-world coordinates (X,Y) of the keys on the table or keyboard
# Units should match your robot/world frame (meters or mm)
world_points = np.array([
    [100, 0],  # key H
    [90, 16.5],  # key G
    [109.5, -16.5],  # key J
    [78.8, -1.2],  # key B
    [121, 1.2],  # key U
    [71.5, 49.4],  # key D
], dtype=np.float32)

# -----------------------------
# Step 2: Compute the Homography
# -----------------------------
H, status = cv2.findHomography(pixel_points, world_points, cv2.RANSAC, 5.0)
print("Homography matrix:\n", H)

# -----------------------------
# Step 3: Function to convert pixel to world coordinates
# -----------------------------
def pixel_to_world(u, v, H):
    pixel_h = np.array([u, v, 1], dtype=np.float32)  # homogeneous coordinates
    world_h = H @ pixel_h
    X = world_h[0] / world_h[2]
    Y = world_h[1] / world_h[2]
    return X, Y

# -----------------------------
# Step 4: Example usage
# -----------------------------
u, v = 503, 256  # some pixel coordinate
X, Y = pixel_to_world(u, v, H)
print(f"Pixel ({u}, {v}) -> World coordinates: X={X:.3f} m, Y={Y:.3f} m")