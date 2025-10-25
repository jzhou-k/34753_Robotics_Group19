#!/usr/bin/env python3
from visual_kinematics.RobotSerial import RobotSerial
import numpy as np
import math


def set_equal_aspect(ax, pts):
    """Make axes equal around the data points."""
    pts = np.asarray(pts, dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    if max_range <= 0:
        max_range = 1.0
    mid = np.array([x.mean(), y.mean(), z.mean()], dtype=float)
    r = 0.6 * max_range
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)

def plot_robot_4link(angles, dh_table, title="Robot (4 links)", draw_axes=True):
    """
    Draw a 4-link serial arm using your Standard-DH convention.
    - angles: iterable of 4 joint angles [t1..t4] in radians
    - dh_table: array-like with 4 rows of [theta_ref, d, a, alpha].  theta_ref is a
      reference/offset; we add 'angles[i]' to it when building each transform.
    """
    import matplotlib.pyplot as plt

    angles = np.asarray(angles, dtype=float)
    dh_table = np.asarray(dh_table, dtype=float)
    assert dh_table.shape[0] >= 4, "Need at least 4 DH rows."
    assert angles.shape[0] >= 4, "Need 4 joint angles."

    T = np.eye(4)
    origins = [T[:3, 3].copy()]  # O0
    frames = [T.copy()]          # store frames if we want to draw axes

    # Build cumulative transforms using your dh_std(theta, d, a, alpha)
    for i in range(4):
        theta_ref, d, a, alpha = dh_table[i]
        theta_i = theta_ref + angles[i]     # include any constant offset if present
        T = T @ dh_std(theta_i, d, a, alpha)
        origins.append(T[:3, 3].copy())
        frames.append(T.copy())

    origins = np.vstack(origins)  # shape (5,3): O0..O4

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], marker='o', linewidth=2)

    # Label joints
    for idx, p in enumerate(origins):
        ax.text(p[0], p[1], p[2], f"O{idx}")

    # Optional: draw small coordinate axes at each frame
    if draw_axes:
        L = max(1e-9, 0.15 * np.max(np.ptp(origins, axis=0)))  # axis length
        for F in frames:
            o = F[:3, 3]
            Rx = F[:3, 0]; Ry = F[:3, 1]; Rz = F[:3, 2]
            ax.plot([o[0], o[0] + L*Rx[0]], [o[1], o[1] + L*Rx[1]], [o[2], o[2] + L*Rx[2]])
            ax.plot([o[0], o[0] + L*Ry[0]], [o[1], o[1] + L*Ry[1]], [o[2], o[2] + L*Ry[2]])
            ax.plot([o[0], o[0] + L*Rz[0]], [o[1], o[1] + L*Rz[1]], [o[2], o[2] + L*Rz[2]])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title + f"\nAngles (deg): {np.round(np.degrees(angles[:4]), 3)}")
    set_equal_aspect(ax, origins)
    plt.tight_layout()
    plt.show()

# ---------- Geometry / DH utilities ----------

def dh_std(theta, d, a, alpha):
    """
    Standard DH: Rot(z, theta) * Trans(z, d) * Trans(x, a) * Rot(x, alpha)
    Returns a 4x4 NumPy array.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ ct,   -st*ca,  st*sa,  a*ct],
        [ st,    ct*ca, -ct*sa,  a*st],
        [ 0.0,       sa,    ca,    d],
        [ 0.0,      0.0,   0.0,  1.0]
    ], dtype=float)

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

# ---------- Forward kinematics (for verification) ----------
def fk_T04(q, d1, d2, d3, d4):
    t1, t2, t3, t4 = q
    T01 = dh_std(t1, d1,   0.0,  np.pi/2)
    T12 = dh_std(t2, 0.0,  d2,   0.0)
    T23 = dh_std(t3, 0.0,  d3,   0.0)
    T34 = dh_std(t4, 0.0,  d4,   0.0)
    return (((T01 @ T12) @ T23) @ T34)

# ---------- Inverse kinematics ----------
def inverse_ik_from_pose(T04, d1, d2, d3, d4):
    """
    Solve q = [t1,t2,t3,t4] for the 4-DOF arm in your notes.

    Inputs
    ------
    T04 : (4,4) numpy array        # desired pose of frame {4} in {0}
    d1,d2,d3,d4 : floats            # DH parameters exactly as used in your notes
                                    # (a2=d2, a3=d3, a4=d4; d1 is offset on z1)

    Returns
    -------
    q : numpy array shape (4,) in radians
    """
    # Extract orientation (R04) and position (o4) from the target pose
    R04 = T04[0:3, 0:3]
    x4, y4, z4 = T04[0:3, 3]

    # 1) theta1 from plan view
    t1 = math.atan2(y4, x4)

    # 2) theta_total from the first column of R04 (x-axis of frame 4 in {0})
    #    x4^0 = [r11, r21, r31]^T = [cos t1 cos T,  sin t1 cos T,  sin T]^T
    r11, r21, r31 = R04[0, 0], R04[1, 0], R04[2, 0]
    theta_total = math.atan2(r31, math.hypot(r11, r21))  # numerically robust

    # 3) Wrist-centre O3^0 by moving back along x4 by distance d4
    #    O3 = O4 - d4 * (first column of R04)
    x3 = x4 - d4 * r11
    y3 = y4 - d4 * r21
    z3 = z4 - d4 * r31

    # 4) Reduce to 2R planar triangle (links a2=d2, a3=d3)
    r = math.hypot(x3, y3)           # horizontal distance
    s = z3 - d1                       # vertical distance measured from joint-2 plane

    # 5) theta3 from law of cosines
    cos_t3 = clamp((r*r + s*s - d2*d2 - d3*d3) / (2.0*d2*d3))
    # Choose elbow-down: negative sin; use +sqrt for elbow-up
    sin_t3 = -math.sqrt(max(0.0, 1.0 - cos_t3*cos_t3))
    t3 = math.atan2(sin_t3, cos_t3)

    # 6) theta2 from two-triangle rule
    t2 = math.atan2(s, r) - math.atan2(d3*sin_t3, d2 + d3*cos_t3)

    # 7) theta4 from theta_total
    t4 = theta_total - t2 - t3

    return np.array([t1, t2, t3, t4], dtype=float)

def main():
    # -----------------------------
    # 1) Robot geometry (mm)
    # -----------------------------
    d1 = 50.0
    d2 = 93.0
    d3 = 93.0
    d4 = 50.0

    # Standard DH table: [theta_ref, d, a, alpha]
    DH_table = np.array([
        [0.0,  d1, 0.0,  np.pi/2],  # Joint 1
        [0.0, 0.0,  d2,  0.0],      # Joint 2
        [0.0, 0.0,  d3,  0.0],      # Joint 3
        [0.0, 0.0,  d4,  0.0],      # Joint 4
    ], dtype=float)

    # -----------------------------
    # 2) User inputs
    # -----------------------------
    pos4 = np.array([120.0, 0.0, 120.0], dtype=float)   # End-effector position (o4^0)
    Rot4_X4_Z0 = 0.0                                       # Last component of x4^0 (z-comp of frame 4 x-axis)

    # -----------------------------
    # 3) Compute theta_total and construct R04
    # -----------------------------
    theta_total = math.asin(Rot4_X4_Z0)
    t1 = math.atan2(pos4[1], pos4[0])

    # Construct x4-axis (first column of R04)
    x4_axis = np.array([
        math.cos(t1) * math.cos(theta_total),
        math.sin(t1) * math.cos(theta_total),
        Rot4_X4_Z0
    ], dtype=float)

    # Complete R04 to a valid rotation matrix
    z_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(x4_axis, z_ref)) > 0.99:
        z_ref = np.array([0.0, 1.0, 0.0])
    y4_axis = np.cross(z_ref, x4_axis); y4_axis /= np.linalg.norm(y4_axis)
    z4_axis = np.cross(x4_axis, y4_axis); z4_axis /= np.linalg.norm(z4_axis)
    R04 = np.column_stack([x4_axis, y4_axis, z4_axis])

    # Assemble T04
    T04 = np.eye(4)
    T04[0:3, 0:3] = R04
    T04[0:3, 3] = pos4

    # -----------------------------
    # 4) Inverse kinematics
    # -----------------------------
    q = inverse_ik_from_pose(T04, d1, d2, d3, d4)
    print("IK solution (rad):", q)
    print("IK solution (deg):", np.rad2deg(q))

    # -----------------------------
    # 5) Forward kinematics check
    # -----------------------------
    T04_fk = fk_T04(q, d1, d2, d3, d4)
    p_fk = T04_fk[0:3, 3]
    print("FK end-effector position (mm):", p_fk)
    print("Target position (mm):         ", pos4)
    print("Position error norm (mm):     ", np.linalg.norm(p_fk - pos4))

    # -----------------------------
    # 6) Visualisation
    # -----------------------------
    plot_robot_4link(q, DH_table, title="IK Solution (using Rot4_X4_Z0 input)")

if __name__ == "__main__":
    main()
