import numpy as np
import sympy as sp


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
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

    return T