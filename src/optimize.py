"""This script carries out optimization using bundle adjustment
"""
from src.utils import *
from src.visualize import *
import matplotlib.pyplot as plt
from src.bundle_adjustment import compute_reprojection_error_total, optimize_using_BA


def optimize_pose_and_map(frames, map_):
    avg_error = compute_reprojection_error_total(frames, map_)
    visualise_pose_and_3d_points(
        [frame.RT for frame in frames], map_.X[:, :3], map_.color
    )
    print("Raw Reprojection Error:", avg_error)
    optimize_using_BA(frames, map_)
    visualise_pose_and_3d_points(
        [frame.RT for frame in frames], map_.X[:, :3], map_.color
    )
    avg_error = compute_reprojection_error_total(frames, map_)
    print("Optimized Reprojection Error:", avg_error)
