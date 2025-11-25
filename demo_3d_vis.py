import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# 1. Load joint angle data
# ----------------------------
# Assumes joint_positions.csv is in the current working directory and has header:
# P1,P2,P3,P4
df = pd.read_csv("joint_positions.csv")
joint_angles_deg = df[["P1", "P2", "P3", "P4"]].values  # shape (N, 4)

# ----------------------------
# 2. DH parameters (same as RobotController)
# ----------------------------
# [d, a, alpha, theta_offset]  (theta_offset not used here, all zeros)
dh_params = np.array([
    [50.0, 0.0, -0.5 * math.pi, 0.0],   # joint 1
    [0.0, 93.0, 0.0, 0.0],              # joint 2
    [0.0, 93.0, 0.0, 0.0],              # joint 3
    [0.0, 50.0, 0.0, 0.0],              # joint 4 (tool)
])

def make_DH_matrix(d, a, alpha, theta):
    """Single DH transform, identical to RobotController.make_DH_matrix."""
    return np.array([
        [ math.cos(theta), -math.sin(theta)*math.cos(alpha),  math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
        [ math.sin(theta),  math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
        [ 0.0,              math.sin(alpha),                  math.cos(alpha),                 d],
        [ 0.0,              0.0,                              0.0,                             1.0]
    ])

def forward_kinematics(joint_angles_deg):
    """
    Compute end-effector position [x, y, z] from four joint angles in degrees.
    Matches RobotController.forward (returns last joint position).
    """
    T = np.eye(4)
    for i, angle_deg in enumerate(joint_angles_deg):
        theta = math.radians(angle_deg)  # deg -> rad
        d, a, alpha, _ = dh_params[i]
        A_i = make_DH_matrix(d, a, alpha, theta)
        T = T @ A_i
    # end-effector position is translation part of T
    return T[0:3, 3]

# ----------------------------
# 3. Compute end-effector path
# ----------------------------
ee_positions = np.array([forward_kinematics(ja) for ja in joint_angles_deg])
xs, ys, zs = ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2]

# ----------------------------
# 4. Plot 3D path
# ----------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot(xs, ys, zs, "-o", markersize=2, linewidth=2)

# mark start and end
ax.scatter(xs[0], ys[0], zs[0], color="green", s=50, label="Start")
ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=50, label="End")

ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")
ax.set_title("End-effector path from joint_positions.csv")
ax.legend()

# make axes roughly equal for a good perspective
max_range = np.array([xs.max()-xs.min(),
                      ys.max()-ys.min(),
                      zs.max()-zs.min()]).max() / 2.0

mid_x = (xs.max()+xs.min()) * 0.5
mid_y = (ys.max()+ys.min()) * 0.5
mid_z = (zs.max()+zs.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# choose a nice viewing angle
ax.view_init(elev=35, azim=-60)

plt.tight_layout()
plt.show()
