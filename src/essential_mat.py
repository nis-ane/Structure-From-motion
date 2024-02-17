"""
This module contains code for
1. Estimation of essential matrix
2. Decompositon of Essential Matrix to R and T
3. Resolve the ambiguity of R and T decomposed from Essential Matrix
"""

import cv2
import numpy as np
import json
from src.utils import get_correspondence_from_file
from src.utils import Vec2Skew
from scipy.spatial.distance import cdist

# Load 2D correspondences
correspondences_file = '/Users/hewanshrestha/Desktop/3dcv_project/Stage_1/submission/box/correspondences/0_2.txt'
points1, points2 = get_correspondence_from_file(correspondences_file)

# Load camera intrinsics from the JSON file
json_file = '/Users/hewanshrestha/Desktop/3dcv_project/Stage_1/submission/box/gt_camera_parameters.json'
with open(json_file, 'r') as f:
    intrinsics_data = json.load(f)
K = np.array(intrinsics_data['intrinsics'])


def normalize_points(points, intrinsics):
    # Use only the first two dimensions of points
    points = points[:, :2]

    # Append ones to the points for homogeneous coordinates
    points_homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)

    # Apply inverse intrinsics to normalize
    normalized_points_homogeneous = np.dot(points_homogeneous, np.linalg.inv(intrinsics).T)

    # Extract normalized coordinates
    normalized_points = normalized_points_homogeneous[:, :2] / normalized_points_homogeneous[:, 2, np.newaxis]

    return normalized_points


# Normalize 2D points
points1_normalized = normalize_points(points1, K)
points2_normalized = normalize_points(points2, K)

# # Calculate essential matrix
E2, mask = cv2.findEssentialMat(points1_normalized, points2_normalized, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)


def calculate_essential_matrix(points1, points2):

    points1 = normalize_points(points1, K)
    points2 = normalize_points(points2, K)

    # Convert to homogeneous coordinates
    points1 = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2 = np.hstack((points2, np.ones((points2.shape[0], 1))))

    # Construct matrix A using the Kronecker product
    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        A[i] = np.kron(points1[i][:3], points2[i][:3])

    # Solve the equation Af = 0 using SVD
    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)

    # Enforce the rank-2 constraint on F
    U, S, Vt = np.linalg.svd(E)
    # S[2] = 0
    E = np.dot(U, np.dot(np.diag([1,1,0]), Vt))

    return E


def calculate_essential_matrix_ransac(points1, points2, iterations=1000, threshold=1e-3):
    best_inliers = 0
    best_E = None

    for _ in range(iterations):
        # Randomly select 8 point correspondences
        indices = np.random.choice(len(points1), 8, replace=False)
        points1_sample = points1[indices]
        points2_sample = points2[indices]

        # Calculate the essential matrix from the sample
        E_sample = calculate_essential_matrix(points1_sample, points2_sample)

        # Calculate the Sampson distance
        F_points1 = np.dot(E_sample, points1.T).T
        F_points2 = np.dot(E_sample.T, points2.T).T
        denom = F_points1[:, 0]**2 + F_points1[:, 1]**2 + F_points2[:, 0]**2 + F_points2[:, 1]**2
        sampson_distance = (np.sum(points1 * F_points1, axis=1)**2) / denom

        # Find inliers
        inliers = sampson_distance < threshold

        # If the number of inliers is higher than the previous best, update the best essential matrix
        if np.sum(inliers) > best_inliers:
            best_inliers = np.sum(inliers)
            best_E = E_sample

    return best_E


def calculate_normalized_essential_matrix(points1, points2, K):
    # Normalize the points
    points1_normalized = normalize_points(points1, K)
    points2_normalized = normalize_points(points2, K)

    # Convert to homogeneous coordinates
    points1_normalized = np.hstack((points1_normalized, np.ones((points1_normalized.shape[0], 1))))
    points2_normalized = np.hstack((points2_normalized, np.ones((points2_normalized.shape[0], 1))))

    # Calculate the transformation matrices
    T1 = calculate_transformation_matrix(points1_normalized)
    T2 = calculate_transformation_matrix(points2_normalized)

    # Transform the points
    points1_transformed = np.dot(T1, points1_normalized.T).T
    points2_transformed = np.dot(T2, points2_normalized.T).T

    # Calculate the essential matrix from the transformed points
    E_transformed = calculate_essential_matrix_ransac(points1_transformed, points2_transformed)

    # Denormalize the essential matrix
    E = np.dot(T1.T, np.dot(E_transformed, T2))

    return E

#
def calculate_transformation_matrix(points):
    # Calculate the center of mass of the points
    center = np.mean(points, axis=0)

    # Translate the points so that the center of mass is at (0, 0)
    points_centered = points - center

    # Calculate the scaling factor
    scale = np.max(np.abs(points_centered))

    # Construct the transformation matrix
    T = np.array([[1/scale, 0, -center[0]/scale],
                  [0, 1/scale, -center[1]/scale],
                  [0, 0, 1]])

    return T

# Calculate the essential matrix
E = calculate_normalized_essential_matrix(points1, points2, K)

# Decompose the essential matrix into R and T
def decompose_essential_matrix(E):
    # Compute SVD of E
    U, S, Vt = np.linalg.svd(E)

    # Ensure it's a right-handed coordinate system
    if np.linalg.det(np.dot(U, Vt)) < 0:
        Vt = -Vt

    # Compute rotation matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = np.dot(np.dot(U, W), Vt)

    # Compute translation vector (up to scale)
    t = U[:, 2]

    # Normalize the translation vector to have unit norm
    t /= np.linalg.norm(t)

    # Reshape t to be a column vector
    t = t.reshape(3, 1)

    return R, t


R, t = decompose_essential_matrix(E)
if not np.allclose(R.T, np.linalg.inv(R)):
    print("R is not orthogonal")
else:
    print("R is orthogonal")

# Check determinant of R
if not np.isclose(np.linalg.det(R), 1):
    print("Determinant of R is not 1")
else:
    print("Determinant of R is 1")


print("R:", R)
print(R.shape)
print("\nt:", t)
print(t.shape)

RT = np.hstack((R.T, t))
print("\nRT:", RT)


_, R_cv, t_cv, mask = cv2.recoverPose(E, points1_normalized, points2_normalized, K)
print("\nR_cv:", R_cv)
print(R_cv.shape)
print("\nt_cv:", t_cv)
print(t_cv.shape)

# Output the essential matrix, rotation, and translation
print("\nEssential Matrix:")
print(E)

print("\nEssential Matrix(cv2):")
print(E2)

# R1, R2, t = decompose_essential_matrix(E)
# # Print the results
# print("\nDecomposed Rotation Matrix R1:")
# print(R1)

# print("\nDecomposed Rotation Matrix R2:")
# print(R2)

# print("\nDecomposed Translation Vector t:")
# print(t)


# import numpy as np

# # Decompose the essential matrices
# R1_1, R1_2, t1 = decompose_essential_matrix(E)
# R2_1, R2_2, t2 = decompose_essential_matrix(E2)

# # Calculate the angles between the rotation matrices
# angle1 = np.degrees(np.arccos((np.trace(np.dot(R1_1.T, R2_1)) - 1) / 2))
# angle2 = np.degrees(np.arccos((np.trace(np.dot(R1_1.T, R2_2)) - 1) / 2))
# angle3 = np.degrees(np.arccos((np.trace(np.dot(R1_2.T, R2_1)) - 1) / 2))
# angle4 = np.degrees(np.arccos((np.trace(np.dot(R1_2.T, R2_2)) - 1) / 2))

# # Calculate the angles between the translation vectors
# angle_t = np.degrees(np.arccos(np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))))

# # Print the angles
# print("Angles between rotation matrices (in degrees): ", angle1, angle2, angle3, angle4)
# print("Angle between translation vectors (in degrees): ", angle_t)

# import os

# # Get the list of all correspondence files
# correspondences_dir = '/Users/hewanshrestha/Desktop/3dcv_project/Stage_1/submission/box/correspondences/'
# correspondences_files = [os.path.join(correspondences_dir, f) for f in os.listdir(correspondences_dir) if f.endswith('.txt')]

# # Process each file
# for correspondences_file in correspondences_files:
#     points1, points2 = get_correspondence_from_file(correspondences_file)

#     # Normalize 2D points
#     points1_normalized = normalize_points(points1, K)
#     points2_normalized = normalize_points(points2, K)

#     # Calculate essential matrix
#     E, mask = cv2.findEssentialMat(points1_normalized, points2_normalized, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

#     # Recover pose from essential matrix
#     points, R, t, mask = cv2.recoverPose(E, points1_normalized, points2_normalized, K)

#     # Output the essential matrix, rotation, and translation
#     print(f"For file {correspondences_file}:")
#     print("Essential Matrix:")
#     print(E)
#     print("\nRotation Matrix:")
#     print(R)
#     print("\nTranslation Vector:")
#     print(t)
#     print("\n------------------------\n")

