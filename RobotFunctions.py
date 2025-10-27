#!/usr/bin/env python3
# RobotFunctions.py

import numpy as np
import sympy as sp
import math

# =========================
# SymPy DH utilities (yours)
# =========================

def make_DH_matrix(theta, d, a, alpha):
    """
    Standard DH: Rot(z,theta) * Trans(z,d) * Trans(x,a) * Rot(x,alpha)
    Returns a 4x4 sympy.Matrix (works with symbols or numbers).
    """
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)

    return sp.Matrix([
        [ ct,    -st*ca,   st*sa,  a*ct],
        [ st,     ct*ca,  -ct*sa,  a*st],
        [  0,         sa,      ca,    d],
        [  0,          0,       0,    1]
    ])

def print_matrix_aligned(name, M, digits=3):
    """
    Pretty, column-aligned printing for SymPy matrices.
    - Rounds to 'digits' significant digits.
    - Draws column separators.
    """
    Mnum = M.evalf(digits)
    rows, cols = Mnum.rows, Mnum.cols
    s = [[str(Mnum[i, j]) for j in range(cols)] for i in range(rows)]
    col_w = [max(len(s[i][j]) for i in range(rows)) for j in range(cols)]

    print(f"\n{name} =")
    top = "┌ " + " │ ".join(" " * w for w in col_w) + " ┐"
    bot = "└ " + " │ ".join(" " * w for w in col_w) + " ┘"
    print(top)
    for i in range(rows):
        parts = [s[i][j].rjust(col_w[j]) for j in range(cols)]
        print("│ " + " │ ".join(parts) + " │")
    print(bot)

# =========================
# NumPy DH + robot functions
# =========================

def dh_std_np(theta, d, a, alpha):
    """
    Standard DH matrix as NumPy array:
    Rot(z,theta) * Trans(z,d) * Trans(x,a) * Rot(x,alpha)
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

def set_equal_aspect(ax, pts):
    """
    Make axes equal around the data points.
    pts: (N,3) array-like
    """
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
    Draw a 4-link serial arm using Standard-DH.
    - angles: iterable of 4 joint angles [t1..t4] in radians
    - dh_table: 4 rows of [theta_ref, d, a, alpha] (theta_ref is offset)
    """
    import matplotlib.pyplot as plt

    angles = np.asarray(angles, dtype=float)
    dh_table = np.asarray(dh_table, dtype=float)
    assert dh_table.shape[0] >= 4, "Need at least 4 DH rows."
    assert angles.shape[0] >= 4, "Need 4 joint angles."

    T = np.eye(4)
    origins = [T[:3, 3].copy()]  # O0
    frames = [T.copy()]

    for i in range(4):
        theta_ref, d, a, alpha = dh_table[i]
        theta_i = theta_ref + angles[i]
        T = T @ dh_std_np(theta_i, d, a, alpha)
        origins.append(T[:3, 3].copy())
        frames.append(T.copy())

    origins = np.vstack(origins)  # (5,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], marker='o', linewidth=2)

    # Labels
    for idx, p in enumerate(origins):
        ax.text(p[0], p[1], p[2], f"O{idx}")

    # Frame axes
    if draw_axes:
        L = max(1e-9, 0.15 * np.max(np.ptp(origins, axis=0)))
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

# ---------- FK specific to your 4-DOF geometry ----------
def fk_T04(q, d1, d2, d3, d4):
    """
    Forward kinematics for the specific 4-DOF chain:
    T01: (theta1, d1, 0,   +pi/2)
    T12: (theta2,  0, d2,  0)
    T23: (theta3,  0, d3,  0)
    T34: (theta4,  0, d4,  0)
    """
    t1, t2, t3, t4 = q
    T01 = dh_std_np(t1, d1,  0.0, np.pi/2)
    T12 = dh_std_np(t2, 0.0, d2,  0.0)
    T23 = dh_std_np(t3, 0.0, d3,  0.0)
    T34 = dh_std_np(t4, 0.0, d4,  0.0)
    return (((T01 @ T12) @ T23) @ T34)

# ---------- Build T04 from inputs (pos4, Rot4_X4_Z0) ----------
def build_pose_from_inputs(pos4_xyz, Rot4_X4_Z0):
    """
    Given:
      - pos4_xyz: [x4, y4, z4] (end-effector position in {0})
      - Rot4_X4_Z0: r31 component = z-projection of x4-axis (first column of R04)
    Construct a rotation R04 consistent with theta_total and t1, and assemble T04.
    """
    pos4 = np.asarray(pos4_xyz, dtype=float).reshape(3,)
    r31 = float(Rot4_X4_Z0)

    # theta_total determined by r31 = sin(theta_total)
    theta_total = math.asin(clamp(r31, -1.0, 1.0))

    # t1 from plan view
    t1 = math.atan2(pos4[1], pos4[0])

    # First column of R04 (x4-axis in {0})
    x4_axis = np.array([
        math.cos(t1) * math.cos(theta_total),
        math.sin(t1) * math.cos(theta_total),
        r31
    ], dtype=float)

    # Complete a right-handed R04
    z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(x4_axis, z_ref)) > 0.99:
        z_ref = np.array([0.0, 1.0, 0.0], dtype=float)

    y4_axis = np.cross(z_ref, x4_axis)
    y4_axis /= np.linalg.norm(y4_axis)
    z4_axis = np.cross(x4_axis, y4_axis)
    z4_axis /= np.linalg.norm(z4_axis)

    R04 = np.column_stack([x4_axis, y4_axis, z4_axis])

    T04 = np.eye(4)
    T04[0:3, 0:3] = R04
    T04[0:3, 3] = pos4
    return T04

# ---------- Inverse kinematics ----------
def inverse_ik_from_pose(T04, d1, d2, d3, d4):
    """
    Solve q = [t1, t2, t3, t4] for the 4-DOF arm.

    T04: (4,4) numpy array
    d1,d2,d3,d4: link parameters (a2=d2, a3=d3, a4=d4; d1 is base offset along z1)
    """
    R04 = T04[0:3, 0:3]
    x4, y4, z4 = T04[0:3, 3]

    # 1) theta1
    t1 = math.atan2(y4, x4)

    # 2) theta_total from x4-axis in {0} = first column of R04
    r11, r21, r31 = R04[0, 0], R04[1, 0], R04[2, 0]
    theta_total = math.atan2(r31, math.hypot(r11, r21))

    # 3) Wrist-centre O3 = O4 - d4 * x4_axis
    x3 = x4 - d4 * r11
    y3 = y4 - d4 * r21
    z3 = z4 - d4 * r31

    # 4) Planar reduction
    r = math.hypot(x3, y3)
    s = z3 - d1

    # 5) theta3 from cosine law
    cos_t3 = clamp((r*r + s*s - d2*d2 - d3*d3) / (2.0*d2*d3))
    # Elbow-up (positive sin) by default; change sign for elbow-down
    sin_t3 = +math.sqrt(max(0.0, 1.0 - cos_t3*cos_t3))
    t3 = math.atan2(sin_t3, cos_t3)

    # 6) theta2
    t2 = math.atan2(s, r) - math.atan2(d3*sin_t3, d2 + d3*cos_t3)

    # 7) theta4 from total
    t4 = theta_total - t2 - t3

    return np.array([t1, t2, t3, t4], dtype=float)

# Runs Forward and Inverse Kinematics + Visualisation
def solve_fk_ik_and_visualize(pos4_xyz, Rot4_X4_Z0, d1, d2, d3, d4, draw_axes=True):
    """
    Given end-effector position and Rot4_X4_Z0, solve IK, verify with FK, and visualize.
    Returns (q, T04_fk, T04_target).
    """
    # Build the target pose
    T04_target = build_pose_from_inputs(pos4_xyz, Rot4_X4_Z0)

    # IK
    q = inverse_ik_from_pose(T04_target, d1, d2, d3, d4)

    # FK check
    T04_fk = fk_T04(q, d1, d2, d3, d4)

    # Build DH table for visualization
    DH_table = np.array([
        [0.0,  d1, 0.0,  np.pi/2],  # Joint 1
        [0.0, 0.0,  d2,  0.0],      # Joint 2
        [0.0, 0.0,  d3,  0.0],      # Joint 3
        [0.0, 0.0,  d4,  0.0],      # Joint 4
    ], dtype=float)

    # Visualize
    plot_robot_4link(q, DH_table, title="IK Solution", draw_axes=draw_axes)

    return q, T04_fk, T04_target

def jacobian(q, d, T45=None):
    """
    Geometric Jacobians for your 4-DOF robot using Standard-DH (all revolute).
    Args
    ----
    q   : iterable (t1, t2, t3, t4) in radians
    d   : iterable (d1, d2, d3, d4) link params per your fk_T04 convention
          T01: (theta1,d1, 0, +pi/2)
          T12: (theta2, 0, d2, 0)
          T23: (theta3, 0, d3, 0)
          T34: (theta4, 0, d4, 0)
    T45 : optional 4×4 SymPy transform for camera pose {4}→{5}.
          If None, identity is used (camera at the EE).

    Returns
    -------
    (J4, J5) : 6×4 SymPy matrices evaluated at q
               J4 is EE Jacobian (frame {4}); J5 is camera Jacobian (frame {5})
    """
    t1, t2, t3, t4 = map(sp.nsimplify, q)
    d1, d2, d3, d4 = d
    pi = sp.pi

    # Symbols
    theta1, theta2, theta3, theta4 = sp.symbols('theta1 theta2 theta3 theta4')

    # Per-link DH transforms (symbolic)
    T01 = make_DH_matrix(theta1, d1,  0,  pi/2)
    T12 = make_DH_matrix(theta2, 0,   d2, 0)
    T23 = make_DH_matrix(theta3, 0,   d3, 0)
    T34 = make_DH_matrix(theta4, 0,   d4, 0)

    # Cumulative
    T02 = T01 * T12
    T03 = T02 * T23
    T04 = T03 * T34
    if T45 is None:
        T45 = sp.eye(4)
    T05 = T04 * T45

    # Axes z_{i-1} and origins O_{i-1} in {0}
    z0 = sp.Matrix([0, 0, 1]); O0 = sp.Matrix([0, 0, 0])
    z1, O1 = T01[0:3, 2], T01[0:3, 3]
    z2, O2 = T02[0:3, 2], T02[0:3, 3]
    z3, O3 = T03[0:3, 2], T03[0:3, 3]
    O4      = T04[0:3, 3]
    O5      = T05[0:3, 3]

    Z = [z0, z1, z2, z3]
    O = [O0, O1, O2, O3]

    def _J_at(Ok):
        cols_v = [Z[i].cross(Ok - O[i]) for i in range(4)]
        cols_w = [Z[i] for i in range(4)]
        Jv = sp.Matrix.hstack(*cols_v)
        Jw = sp.Matrix.hstack(*cols_w)
        J  = sp.Matrix.vstack(Jv, Jw)
        return J.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4}).applyfunc(
            lambda x: 0 if abs(sp.N(x)) < 1e-8 else sp.N(x, 6)
        )

    J4 = _J_at(O4)
    J5 = _J_at(O5)
    return J4, J5