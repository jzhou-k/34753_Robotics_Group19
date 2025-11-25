import cv2
import easyocr #pip install easyocr
import math
import numpy as np
from keydetector import *

#function to pass keyboard h key position and the rotatoin, returns the coordinates of rest of the keys
def keyboard_key_positions_mm(x_h, y_h, rotation=0.0, scale=0.95):

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

    key_height_offset = 1#mm
    row_z_offsets = {
        0: key_height_offset ,  # top row
        1: 0 ,  # home row
        2: -key_height_offset   # bottom row (fix)
    }

    cos_r, sin_r = math.cos(-rotation), math.sin(-rotation)

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

            positions[key] = (x_h + dx_r, y_h + dy_r, row_z_offsets[i])

    bottom_row = 2
    space_row_relative_to_H = (bottom_row + 1) - ref_row  # 1 row below bottom row
    space_dx = -(space_row_relative_to_H * key_pitch)     # dx = ref_row - i
    space_dy = 0                                         # centered under 'H'

    dx_r, dy_r = rotate(space_dx, space_dy)
    positions[' '] = (x_h + dx_r, y_h + dy_r, -key_height_offset*2)

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

def pixel_to_world_coordinates_homogrphy(hx,hy):
    pixels = np.array([[ 45.5, 306.5],[452.0, 368.5],[223.0, 396.0],[150.5, 291.5],[105.0, 182.0],[264.0, 279.5],[380.0, 267.5],
                    [493.5, 254.5],[677.5, 124.5],[607.0, 243.0],[717.5, 232.0],[828.5, 219.5],[674.5, 345.0],[564.0, 356.0],
                    [790.0, 114.0],[903.5, 107.5],[219.0, 171.0],[ 45.0, 306.0],[337.5, 158.0],[565.5, 139.5],[338.0, 382.5],
                    [ 22.5, 195.5],[110.5, 409.0],[ 25.0, 422.5],[453.5, 146.5]], dtype=np.float32)

    worlds = np.array([[105.0,  97.0],[ 93.0,  13.0],[ 90.0,  51.0],[108.0,  60.0],[126.0,  66.0],[109.0,  41.0],[109.0,  22.0],
                    [112.0,   4.0],[133.0, -23.0],[116.0, -10.0],[117.0, -32.0],[118.0, -50.0],[ 98.0, -23.0],[ 96.0,  -6.0],
                    [137.0, -42.0],[138.0, -61.0],[129.0,  49.0],[107.0,  79.0],[129.0,  30.0],[132.0,  -8.0],[ 91.0,  33.0],
                    [124.0,  84.0],[ 90.0,  65.0],[132.0,  12.0],[ 87.0,  84.0]], dtype=np.float32)

    H, _ = cv2.findHomography(pixels, worlds, method=cv2.RANSAC)

    #to get the world coordinates
    pixel = np.array([hx, hy, 1.0])
    world = H @ pixel
    world /= world[2]  # normalize
    X, Y = world[0], world[1]
    return X,Y

#manual calculations - not very accurate
def image_to_world_coordinates(img,x,y,z,u,v,pixel_span=1225, real_span=140):
    #total distance from image to camera

    img_h, img_w = img.shape[:2]
    l = img_w * (real_span / pixel_span)
    
    u_norm = (u / img_w) - 0.5
    v_norm = (v / img_h) - 0.3
    
    wx = v_norm * l
    wy = u_norm * l
    wz = z    

    return wx, wy, wz
