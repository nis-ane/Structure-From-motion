import numpy as np


def Vec2Skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def get_correspondence_from_file(file):
    f = open(file, "r")
    pts_1 = []
    pts_2 = []
    for x in f:
        correspondence = [float(coor) for coor in x.split(" ")]
        pts_1.append((correspondence[0], correspondence[1]))
        pts_2.append((correspondence[2], correspondence[3]))
    return pts_1, pts_2
