import numpy as np

def Vec2Skew(v):
    return np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]], [-v[1], v[0], 0]])

