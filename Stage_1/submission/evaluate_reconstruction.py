import numpy as np
from scipy.spatial import KDTree
import trimesh


def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)


pcd_gt = trimesh.load("box/gt_points.ply")
gt_points = np.array(pcd_gt.vertices, dtype=np.float32)

pcd_test = trimesh.load(
    "box/estimated_points.ply"
)  # Give path to your .ply here (as above)
test_points = np.array(pcd_test.vertices, dtype=np.float32)

print("Chamfer distance between pointclouds:", chamfer_distance(gt_points, test_points))
