from src.utils import *
from src.visualize import *


def optimize_pose_and_map(frames, map_, n):
    compute_reprojection_error(frames[n], map_)
    visualize_reprojection_error(frames[n], map_)


def compute_reprojection_error(frame, map_):
    X = map_.X[frame.index_kp_3d]
    x_proj = project_3D_to_2D(X, frame.P)
    x = frame.keypoints[frame.triangulated_idx]
    print("Reprojection error:", np.linalg.norm(x - x_proj))
    assert len(x_proj) == len(x)
