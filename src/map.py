import numpy as np


class Map:
    def __init__(self):
        self.X = np.zeros((0, 4))
        self.color = []

    def update_map(self, X_new, color):
        self.X = np.vstack((self.X, X_new))
        self.color.extend(color)

    def update_coordinates(self, X):
        self.X = X


def register_frames_with_map(frame_prev, frame_curr, map_, X):
    """Associates traingulated 3d point index to the corresponding 2d points

    Args:
        frame_prev (Frame): First frame used for traingulation
        frame_curr (Frame): Second frame used for traingulation
        map_ (Map): map of 3d point clouds
        X (np.array shape(n, 4)): Current set of traingulated points
    """
    frame_prev.triangulated_idx = frame_prev.triangulated_idx + list(
        np.array(frame_prev.matched_idx)[frame_prev.disjoint_idx]
    )
    frame_curr.triangulated_idx = frame_curr.triangulated_idx + frame_curr.matched_idx

    assert len(frame_prev.disjoint_idx) == len(X)
    frame_prev.index_kp_3d = frame_prev.index_kp_3d + list(
        range(len(map_.X), len(map_.X) + len(X))
    )
    frame_curr.index_kp_3d = frame_curr.index_kp_3d + list(
        range(len(map_.X), len(map_.X) + len(X))
    )
