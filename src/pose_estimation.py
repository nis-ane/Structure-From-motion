"""
This script should implement the estimation of initial pose for frame
1. Use Essentail matrix decomposition
2. Linear PnP with 3D-2D correspondence
"""
from src.utils import *
from src.visualize import *
from src.essential_mat import recover_pose_using_Essential_Mat
import numpy as np


def cleanup_RT_mat(R, T):
    u, d, v = np.linalg.svd(R)
    det = np.round(np.linalg.det(np.dot(u, v)))
    if det == 1:
        R_cleaned = np.dot(u, v)
        T_cleaned = T / d[0]
    elif det == -1:
        R_cleaned = -np.dot(u, v)
        T_cleaned = -T / d[0]
    return R_cleaned, T_cleaned


def decompose_projection_mat(P, K):
    RT = np.dot(np.linalg.inv(K), P)
    R = RT[:, 0:3]
    T = RT[:, -1].reshape(3, 1)
    R_c, T_c = cleanup_RT_mat(R, T)
    return R_c, T_c


def estimate_pose_Essential_Matrix(frame_1, frame_2):
    points_1 = frame_1.keypoints[frame_1.matched_idx]
    points_2 = frame_2.keypoints[frame_2.matched_idx]

    R, T = recover_pose_using_Essential_Mat(points_1, points_2, frame_1.K)

    return R, T


def estimate_pose_Linear_PnP(x, X, K):
    assert len(x) == len(X)
    A = np.zeros((0, 12))
    for i in range(len(x)):
        skew1 = Vec2Skew(x[i])
        zeros = np.zeros((1, 4))
        X_tilda = X[i].reshape(1, -1)
        X_mat = np.vstack(
            (
                np.hstack((X_tilda, zeros, zeros)),
                np.hstack((zeros, X_tilda, zeros)),
                np.hstack((zeros, zeros, X_tilda)),
            )
        )
        A = np.vstack((A, np.dot(skew1, X_mat)))

    u, s, v = np.linalg.svd(A)
    P = v.T[:, -1].reshape(3, 4)
    R, T = decompose_projection_mat(P, K)
    return R, T


def estimate_pose_Linear_PnP_n(x, X, K):
    assert len(x) == len(X)
    x_normalized = np.dot(np.linalg.inv(K), np.array(x).T).T
    A = np.zeros((0, 12))
    for i in range(len(x_normalized)):
        skew1 = Vec2Skew(x_normalized[i])
        zeros = np.zeros((1, 4))
        X_tilda = X[i].reshape(1, -1)
        X_mat = np.vstack(
            (
                np.hstack((X_tilda, zeros, zeros)),
                np.hstack((zeros, X_tilda, zeros)),
                np.hstack((zeros, zeros, X_tilda)),
            )
        )
        A = np.vstack((A, np.dot(skew1, X_mat)))

    u, s, v = np.linalg.svd(A)
    RT = v.T[:, -1].reshape(3, 4)
    R = RT[:, 0:3]
    T = RT[:, -1].reshape(3, 1)
    R_c, T_c = cleanup_RT_mat(R, T)
    return R_c, T_c


def estimate_pose_Linear_PnP_RANSAC(x, X, K, threshold=5.0):

    best_inliers = []

    for _ in range(200):
        # Randomly sample minimal subset of correspondences
        sample_indices = np.random.choice(len(x), 6, replace=False)
        sample_x = [x[i] for i in sample_indices]
        sample_X = [X[i] for i in sample_indices]

        # Compute candidate pose using Linear PnP
        candidate_R, candidate_T = estimate_pose_Linear_PnP(sample_x, sample_X, K)
        candidate_RT = np.hstack((candidate_R, candidate_T))
        candidate_P = np.dot(K, candidate_RT)

        # Compute reprojection error and find inliers
        inliers = []
        for i in range(len(x)):
            # Project 3D points onto the image plane using candidate pose
            projected_x = project_3D_to_2D(X[i], candidate_P)
            # Compute reprojection error
            error = np.linalg.norm(projected_x - x[i])
            # Check if error is below threshold
            if error < threshold:
                inliers.append(i)

        # Check if current model has more inliers than the best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Refine pose estimate using all inliers
    refined_R, refined_T = estimate_pose_Linear_PnP(
        [x[i] for i in best_inliers], [X[i] for i in best_inliers], K
    )

    return refined_R, refined_T, len(best_inliers)


def estimate_pose_3d_2d_mapping(map_, frame_curr):
    idx_3d = frame_curr.index_kp_3d
    X = map_.X[idx_3d]
    x = frame_curr.keypoints[np.array(frame_curr.matched_idx)[frame_curr.intersect_idx]]
    R, T = estimate_pose_Linear_PnP(x, X, frame_curr.K)
    return R, T
