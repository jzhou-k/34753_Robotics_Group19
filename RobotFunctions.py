import numpy as np
import sympy as sp
import math

# standalone robot utility functions


def make_DH_matrix(d, a, alpha, theta):
    """
    Creates a Denavit-Hartenberg (DH) transformation matrix.

    Parameters:
        d (float): offset along previous z to the common normal
        a (float): length of the common normal (distance along x)
        theta (float): angle about previous z, from old x to new x
        alpha (float): angle about common normal (x axis)

    Returns:
        numpy.ndarray: 4x4 DH transformation matrix
    """
    T = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),
         sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -
         sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),
         sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

    return T


def make_joint_matrices(dh_params_syms):
    """
    Compute forward kinematics for given joint angles (deg).
    Returns a list of matrices representing the transformation from base to each joint.
    """
    T = []
    T_total = sp.eye(4)
    for d, a, theta, alpha in dh_params_syms:
        T_total = make_DH_matrix(d, a, theta, alpha)
        T.append(T_total)

    T45 = sp.eye(4)
    T45[0, 3] = -15
    T45[1, 3] = 45
    T.append(T45)

    return T


def inverse(xyz, theta=0, dh_params=None):
    """Compute inverse kinematics for given target_position [x,y,z]. Returns joint angles (deg)."""
    x, y, z = xyz.flatten()
    d1 = dh_params[0][0]
    a2 = dh_params[1][1]
    a3 = dh_params[2][1]

    theta1 = math.atan2(y, x)
    r = math.sqrt(x**2 + y**2) - a3 * math.cos(theta)
    s = z - d1 - a3 * math.sin(theta)
    cos3 = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    theta3 = math.atan2(cos3, math.sqrt(1 - cos3**2))
    sin3 = math.sin(theta3)
    theta2 = math.atan2(r, s) - math.atan2(a2 + a3 * cos3, a3 * sin3)
    theta4 = theta - theta2 - theta3
    joint_angles = [math.degrees(ang)
                    for ang in [theta1, theta2, theta3, theta4]]
    return joint_angles
