"""
This script should contain the code for
1. Implementation of triangulation to generate 3D points
"""

from src.utils import Vec2Skew
import numpy as np


def triangulate_pts(x1, x2, P1, P2):
    """_summary_

    Args:
        x1 (np.array, shape(n,3)): 2D homogeneous coordinates from first image
        x2 (np.array, shape(n,3)): 2D homogeneous coordinates from second image
        P1 (np.array, shape(3,4)): Projection matrix of first frame
        P2 (np.array, shape(3,4)): Projection matrix of second frame

    Returns:
        X_h (np.array, shape(n,4)): 3D triangulated homogeneous coordinate in space
    """
    assert len(x1) == len(x2)
    Xh = np.zeros((0, 4), dtype=np.float32)
    for i in range(len(x1)):
        skew1 = Vec2Skew(x1[i])
        skew2 = Vec2Skew(x2[i])
        A = np.vstack((np.dot(skew1, P1), np.dot(skew2, P2)))
        u, s, v = np.linalg.svd(A)
        X = v.T[:, -1]
        Xn = X / X[-1]
        Xh = np.vstack((Xh, Xn), dtype=np.float32)
    return Xh
