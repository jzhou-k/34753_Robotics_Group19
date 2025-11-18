import cv2
# import easyocr #pip install easyocr
import math
import numpy as np
from keydetector import *

def keyboard_key_positions_mm(x_h, y_h, rotation=0.0, scale=1.0):

    layout = [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
    ]

    key_pitch = 19.0 * scale

    # find H reference
    for i, row in enumerate(layout):
        if 'H' in row:
            ref_row, ref_col = i, row.index('H')
            break

    # correct staggering
    row_offsets = {
        0: 0.5 * key_pitch,  # top row
        1: 0.0 * key_pitch,  # home row
        2: -0.5 * key_pitch   # bottom row (fix)
    }

    cos_r, sin_r = math.cos(rotation), math.sin(rotation)

    def rotate(dx, dy):
        x_rot = dx * cos_r - dy * sin_r
        y_rot = dx * sin_r + dy * cos_r
        return x_rot, y_rot

    positions = {}

    for i, row in enumerate(layout):
        stagger = row_offsets[i]
        for j, key in enumerate(row):

            dx = (ref_row - i) * key_pitch
            dy = -(j - ref_col) * key_pitch + stagger

            dx_r, dy_r = rotate(dx, dy)

            positions[key] = (x_h + dx_r, y_h + dy_r, 0.0)

    return positions



def get_keyboard(img):
    #function to read the keyboard location and orientation
    
    centers = detect_ghj("keyboard.png","testout")

    g = centers[0]
    h = centers[1]
    j = centers[2]
  
    dx = j[0] - g[0]
    dy = j[1] - g[1]

    angle = math.atan2(dy, dx)

    return h,angle

def get_key(image_path, key, out_dir):
    _, centers, _ = detect_keyboard_text(image_path=image_path, target_text=key, out_dir=out_dir)
    print(centers)
    return centers[0], 0

def get_MnMs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # edges = cv2.Canny(gray, 20, 200)

    threshold = 225
    threshold_value = 255

    thresh = cv2.threshold(gray, threshold, threshold_value, cv2.THRESH_BINARY_INV)[1]
    mask = thresh.copy()
    mask = cv2.dilate(thresh, None, iterations = 5)
    mask = mask.copy()
    mask = cv2.erode(mask, None, iterations = 5)

    edges = cv2.Canny(mask, 20, 200)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=50,      
        param1=100,         
        param2=40,       
        minRadius=20,       
        maxRadius=100
    )

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colors_img = img.copy()

    result = []

    for (x, y, r) in circles[0]:
        # Create a mask for the circular region
        mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

        # Extract HSV pixels inside the circle
        hsv_pixels = img_hsv[mask == 255]

        # Compute the mean HSV values
        mean_hue = np.mean(hsv_pixels[:, 0])
        mean_sat = np.mean(hsv_pixels[:, 1])
        mean_val = np.mean(hsv_pixels[:, 2])

        # Simple classification based on Hue
        color = "Unknown"
        if 2 <= mean_hue <= 6:
            color = "Red"
        elif 6 <= mean_hue <= 10:
            color = "Brown"
        elif 10 <= mean_hue <= 16:
            color = "Orange"
        elif 17 <= mean_hue <= 35:
            color = "Yellow"
        elif 36 <= mean_hue <= 85:
            color = "Green"
        elif 86 <= mean_hue <= 125:
            color = "Blue"
        elif 126 <= mean_hue <= 160:
            color = "Purple"
        

        # print(f"Circle at ({x},{y}) → {color}")
        result.append([x,y,color])

        # Draw the circle and label it
        cv2.circle(colors_img, (int(x), int(y)), int(r), (0,255,0), 2)
        cv2.putText(colors_img, color, (int(x - r), int(y - r - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        # cv2.imshow(colors_img)
        # cv2.show()
    
    return result

def calibrate_Camera(img,hx,hy,x,y):
    #use the location of the H key to calibrate the camera position
    print("test")


    

def image_to_world_coordinates(img,x,y,z,u,v,l=4,w=1280,h=960):
    #total distance from image to camera

    d = 45+z+15+l
    yt = (150*d)/194
    px = yt/1200 #mm per pixel

    u_c = w/2
    v_c = h/2

    x_mov = (u_c - v)*px
    y_mov = (v_c - u)*px

    img_h, img_w = img.shape[:2]
    
    # Pixel to normalized coordinates [-0.5, 0.5]
    u_norm = (u / img_w) - 0.5
    v_norm = (v / img_h) - 0.5
    
    # Scale normalized coordinates to world units
    wx = x - v_norm * l   # +y in image is -x in world
    wy = y - u_norm * l   # +x in image is -y in world
    wz = z    

    return wx, wy, wz


keys = keyboard_key_positions_mm(100, 0, rotation=math.radians(30), scale=1.0)
print(keys['H'])
print(keys['G'])
print(keys['J'])
print(keys['B'])
print(keys['U'])
print(keys['D'])

print(keys['O'])
'''
import matplotlib.pyplot as plt

# Example key positions (rotated 45° for testing)
keys_rotated = {
    'H': keys['H'],
    'G': keys['G'],
    'J': keys['J'],
    'B': keys['B'],
    'U': keys['U'],
    'A': keys['A'],
    'P': keys['P'],
    'Z': keys['Z']
}

plt.figure(figsize=(8, 10))
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')

# Plot keys
for k, (x, y, z) in keys_rotated.items():
    # Swap axes according to your convention: X=up, Y=left
    x_plot = x
    y_plot = -y  # flip horizontal so positive is left, negative right
    plt.scatter(y_plot, x_plot, s=100, c='blue')
    plt.text(y_plot + 1, x_plot + 1, k, fontsize=12)

# Highlight H (origin)
x0, y0, _ = keys_rotated['H']
plt.scatter(-y0, x0, c='red', s=120, label='H (origin)')

plt.xlabel('Y (mm, +left, -right)')
plt.ylabel('X (mm, +up)')
plt.title('Keyboard Key Positions (Robot Frame)')
plt.legend()
plt.show()'''
