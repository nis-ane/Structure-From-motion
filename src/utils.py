import numpy as np


def Vec2Skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def project_3D_to_2D(X, P):
    x = np.dot(P, X.T)
    x = (x / x[-1]).T
    return x


def get_correspondence_from_file(file):
    f = open(file, "r")
    pts_1 = []
    pts_2 = []
    for x in f:
        correspondence = [float(coor) for coor in x.split(" ")]
        pts_1.append((correspondence[0], correspondence[1], 1))
        pts_2.append((correspondence[2], correspondence[3], 1))
    return np.array(pts_1, dtype=np.float32), np.array(pts_2, dtype=np.float32)
