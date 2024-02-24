"""This script is used for retriving correspondece for a pair of images
1. Given 3d points and 2d correspondence generate 3d-2d correspondence for new frame
"""
import numpy as np
from scipy.spatial import distance


def get_3d_to_2d_correspondence(X, x1_prev, x1_curr, x2_curr):
    assert len(X) == len(
        x1_prev
    ), "for previous correspondence Respective 3D points are not available"
    assert len(x1_curr) == len(x2_curr)
    dist = distance.cdist(x1_prev, x1_curr, "euclidean")
    min_distances1 = np.min(dist, axis=0)
    idx = np.argmin(dist, axis=0)

    mask = min_distances1 == 0
    idx_f = idx[mask]
    x1_curr_f = x1_curr[mask]
    x2_curr_f = x2_curr[mask]
    X_f = X[idx_f]

    assert len(x2_curr_f) == len(X_f)
    return X_f, x2_curr_f
