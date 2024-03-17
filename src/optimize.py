from src.utils import *
from src.visualize import *
import matplotlib.pyplot as plt
from src.bundle_adjustment import compute_reprojection_error_total
from src.jacobian import optimize_using_BA


def optimize_pose_and_map(frames, map_):
    avg_error = compute_reprojection_error_total(frames, map_)
    print("Raw Reprojection Error:", avg_error)
    optimize_using_BA(frames, map_)
    avg_error = compute_reprojection_error_total(frames, map_)
    print("Optimized Reprojection Error:", avg_error)
