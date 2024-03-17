"""
This module contains code for
1. Estimation of essential matrix
2. Decompositon of Essential Matrix to R and T
3. Resolve the ambiguity of R and T decomposed from Essential Matrix
"""


import numpy as np
import os
from src.utils import get_correspondence_from_file
import json
import cv2
from scipy.linalg import svd
from src.triangulation import triangulate_pts
from src.visualize import visualise_pose_and_3d_points
import random


def camera_points(points, intrinsics):
    # Normalize points by multiplying with the inverse of the intrinsic matrix
    camera_points = np.dot(np.linalg.inv(intrinsics), points.T).T

    # Convert back to non-homogeneous coordinates
    camera_points = camera_points / camera_points[:, 2:]

    return camera_points


def calculate_essential_matrix(points1, points2):
    A = np.zeros((len(points1), 9))

    for i in range(len(points1)):
        x, y, _ = points1[i]
        xp, yp, _ = points2[i]
        A[i] = [xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1]

    # Solve the equation Af = 0 using SVD
    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)

    # u, s, v = np.linalg.svd(A)
    # E = v.T[:, -1].reshape(3,3)

    # Enforce the rank-2 constraint on E
    U, _, Vt = np.linalg.svd(E)
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), Vt))

    E = E / np.linalg.norm(E)

    return E


def ransac_essential_matrix(points1, points2, T=10, threshold=1e-3, k_max=1000):
    random.seed(0)
    best_E = None
    best_inliers = []

    # Convert points to homogeneous coordinates
    # points1 = np.concatenate([points1, np.ones((points1.shape[0], 1))], axis=1)
    # points2 = np.concatenate([points2, np.ones((points2.shape[0], 1))], axis=1)

    for _ in range(k_max):
        # Step 1: Randomly select a subset S of n data points
        indices = np.random.choice(len(points1), 8, replace=False)
        subset_points1 = [points1[i] for i in indices]
        subset_points2 = [points2[i] for i in indices]

        # Step 2: Estimate the Essential matrix E from the data points in S
        E_estimate = calculate_essential_matrix(subset_points1, subset_points2)

        # Step 3: Compute the set S* of correspondences in agreement with the Essential matrix up to a threshold
        inliers = []
        for i in range(len(points1)):
            error = np.abs(points2[i].T @ E_estimate @ points1[i])
            if error < threshold:
                inliers.append(i)

        # Step 4: Check if the cardinality of S* is larger than the threshold T
        if len(inliers) >= T:
            # Case (a): accept S* as the set of inliers and use it to recompute a better estimate of the Essential matrix E
            all_inliers = np.array(inliers)
            updated_E = calculate_essential_matrix(
                [points1[i] for i in all_inliers], [points2[i] for i in all_inliers]
            )
            inliers = []
            for i in range(len(points1)):
                error = np.abs(points2[i].T @ updated_E @ points1[i])
                if error < threshold:
                    inliers.append(i)

            if len(inliers) > len(best_inliers):
                best_E = updated_E
                best_inliers = inliers
        else:
            # Case (b): repeat from Step 1
            continue

    return best_E, best_inliers


def decompose_essential_matrix(E):
    # Compute SVD of E
    U, _, Vt = np.linalg.svd(E)

    # Ensure it's a right-handed coordinate system
    if np.linalg.det(np.dot(U, Vt)) < 0:
        Vt = -Vt
    # if np.linalg.det(U) < 0:
    #     U *= -1
    # if np.linalg.det(Vt) < 0:
    #     Vt *= -1

    # Compute rotation matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = np.dot(np.dot(U, W), Vt)
    R2 = np.dot(np.dot(U, W.T), Vt)

    # assert np.linalg.det(R1) == 1
    # assert np.linalg.det(R2) == 1
    # # If the determinant of R is -1, flip the sign of the rotation matrix
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Compute translation vector (up to scale)
    t = U[:, 2]

    # Normalize the translation vector to have unit norm
    t /= np.linalg.norm(t)

    # Reshape t to be a column vector
    t = t.reshape(3, 1)

    # Generate the four camera pose configurations
    c1 = t
    r1 = R1
    c2 = -t
    r2 = R1
    c3 = t
    r3 = R2
    c4 = -t
    r4 = R2

    return [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]


def count_points_in_front_of_both_cameras(x1, x2, K, R, C):
    T1 = np.zeros(
        (3, 1)
    )  # Initialize the translation to be zero. The camera center is in origin
    R1 = np.eye(3)
    RT1 = np.hstack((R1, T1))
    P1 = np.dot(K, RT1)
    RT2 = np.hstack((R, -np.dot(R, C)))
    P2 = np.dot(K, RT2)
    X = triangulate_pts(x1, x2, P1, P2)
    transformed_pts = np.dot(R, (X[:, :3].T - C.reshape(3, 1)))
    mask = (transformed_pts[2, :] > 0) & (X[:, 2] > 0)
    count = np.count_nonzero(mask)
    return count


def recover_pose_using_Essential_Mat(points_1, points_2, K):
    normalised_points_1 = camera_points(points_1, K)
    normalised_points_2 = camera_points(points_2, K)
    E, _ = ransac_essential_matrix(normalised_points_1, normalised_points_2)
    poses = decompose_essential_matrix(E)
    counts = [
        count_points_in_front_of_both_cameras(
            normalised_points_1, normalised_points_2, K, R, C
        )
        for R, C in poses
    ]
    best_pose_index = np.argmax(counts)
    best_pose = poses[best_pose_index]

    best_R, best_C = best_pose
    best_T = -np.dot(best_R, best_C)
    return best_R, best_T


def visualize_results(R, t):
    # Create the projection matrices
    P_R_t = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

    I_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # Calculate projection matrices
    P1 = np.dot(K, I_0)
    P2 = P_R_t

    # Remove the last row of zeros
    P1 = P1[:3, :]
    P2 = P2[:3, :]

    RT1 = np.eye(4)[:3, :]
    RT2 = P_R_t

    # Triangulate points
    X = triangulate_pts(points1, points2, P1, P2)

    X = X[:, :3] / X[:, 3:]

    print("X:", X[2, :])

    # Visualize 3D points and camera poses
    visualise_pose_and_3d_points([RT1, RT2], X[:, :3])
