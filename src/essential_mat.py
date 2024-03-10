import numpy as np
from src.utils import get_correspondence_from_file
import json
import cv2
from scipy.linalg import svd
from src.triangulation import triangulate_pts
from src.visualize import visualise_pose_and_3d_points
import os


def camera_points(points, intrinsics):
    # Normalize points by multiplying with the inverse of the intrinsic matrix
    camera_points = np.dot(np.linalg.inv(intrinsics), points.T).T

    # Convert back to non-homogeneous coordinates
    camera_points = camera_points[:, :2] / camera_points[:, 2:]

    return camera_points

def calculate_essential_matrix(points1, points2):
    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        # Convert points to homogeneous coordinates
        point1_hom = np.append(points1[i], 1)
        # print(f"point1_hom: {point1_hom}")
        point2_hom = np.append(points2[i], 1)
        # print(f"point2_hom: {point2_hom}")
        
        A[i] = np.kron(point1_hom, point2_hom.T)

    # print(f"A: {A}")
    # Solve the equation Af = 0 using SVD
    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)

    # Enforce the rank-2 constraint on E
    U, _, Vt = np.linalg.svd(E)
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), Vt))

    return E



def calculate_essential_matrix_ransac(points1, points2, iterations=1000, threshold=1e-3):
    np.random.seed(0)
    best_inliers = 0
    best_E = None

    for _ in range(iterations):
        # Randomly select 8 point correspondences
        indices = np.random.choice(len(points1), 8, replace=False)
        points1_sample = points1[indices]
        points2_sample = points2[indices]

        # Calculate the essential matrix from the sample
        E_sample = calculate_essential_matrix(points1_sample, points2_sample)

        # Convert points to homogeneous coordinates
        points1_hom = np.vstack((points1.T, np.ones(points1.shape[0])))
        points2_hom = np.vstack((points2.T, np.ones(points2.shape[0])))

        # Calculate the Sampson distance
        F_points1_hom = np.dot(E_sample, points1_hom).T
        F_points2_hom = np.dot(E_sample.T, points2_hom).T

        # Convert back to non-homogeneous coordinates
        F_points1 = F_points1_hom[:, :2]
        F_points2 = F_points2_hom[:, :2]

        denom = F_points1[:, 0]**2 + F_points1[:, 1]**2 + F_points2[:, 0]**2 + F_points2[:, 1]**2
        sampson_distance = (np.sum(points1 * F_points1, axis=1)**2) / denom

        # Find inliers
        inliers = sampson_distance < threshold

        # If the number of inliers is higher than the previous best, update the best essential matrix
        if np.sum(inliers) > best_inliers:
            best_inliers = np.sum(inliers)
            best_E = E_sample

    return best_E


def decompose_essential_matrix(E):
    # Compute SVD of E
    U, _, Vt = np.linalg.svd(E)

    # Ensure it's a right-handed coordinate system
    if np.linalg.det(np.dot(U, Vt)) < 0:
        Vt = -Vt

    # Compute rotation matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = np.dot(np.dot(U, W), Vt)

    # If the determinant of R is -1, flip the sign of the rotation matrix
    if np.linalg.det(R) < 0:
        R = -R

    # Compute translation vector (up to scale)
    t = U[:, 2]

    # Normalize the translation vector to have unit norm
    t /= np.linalg.norm(t)

    # Reshape t to be a column vector
    t = t.reshape(3, 1)

    # Enforce positive depth using cheirality condition
    R, t = enforce_cheirality_condition(R, t)

    return R, t

def enforce_cheirality_condition(R, t):
    max_positive_depth_count = 0
    best_R = R
    best_t = t

    for sign in [1, -1]:  # Try both signs for the translation vector
        t_candidate = sign * t
        C2 = -np.dot(R.T, t_candidate.flatten())

        X = triangulate_pts(points1, points2, np.eye(3, 4), np.hstack((R, t_candidate)))

        X = X[:, :3] / X[:, 3:]

        # Ensure the correct shapes for the dot product
        C1 = np.zeros_like(X)
        depth_condition = np.dot(X - C1, R[2]) > 0

        # Ensure the correct shapes for the dot product
        C2 = np.zeros_like(X)
        depth_condition &= np.dot(X - C2, R[2]) > 0

        positive_depth_count = np.sum(depth_condition)

        if positive_depth_count > max_positive_depth_count:
            max_positive_depth_count = positive_depth_count
            best_R = R
            best_t = t_candidate

    return best_R, best_t

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

    # Count positive and negative depths
    global positive_depth_count
    global negative_depth_count
    positive_depth_count += np.sum(X[2, :] > 0)
    negative_depth_count += np.sum(X[2, :] < 0)

    X = X[:, :3] / X[:, 3:]
    print("X:", X[2, :])

    # Visualize 3D points and camera poses
    # visualise_pose_and_3d_points([RT1, RT2], X[:, :3])




# correspondences_file = '/Users/hewanshrestha/Desktop/3dcv_project/stage1/box/correspondences/73_108.txt'
# intrinsics_file = '/Users/hewanshrestha/Desktop/3dcv_project/stage1/box/gt_camera_parameters.json'

# points1, points2 = get_correspondence_from_file(correspondences_file)
# with open(intrinsics_file, 'r') as f:
#     intrinsics_data = json.load(f)
# K = np.array(intrinsics_data['intrinsics'])

# # print(f"Calibration Matrix(K): {K}")
# print(f"K: {K.shape}")

# # print(f"Points1: {points1}")
# print(f"points1: {points1.shape}")

# # print(f"Points2: {points2}")
# print(f"points2: {points2.shape}")

# # Normalize the points
# normalized_points1 = camera_points(points1, K)
# # print(f"Normalized points1: {normalized_points1}")
# print(f"Normalized points1: {normalized_points1.shape}")

# normalized_points2 = camera_points(points2, K)
# # print(f"Normalized points2: {normalized_points2}")
# print(f"Normalized points2: {normalized_points2.shape}")

# E_cv, mask = cv2.findEssentialMat(normalized_points1, normalized_points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
# print(f"Essential Matrix(cv2):")
# print(E_cv)

# E = calculate_essential_matrix(normalized_points1, normalized_points2)
# print(f"\nEssential Matrix:")
# print(E)

# # Assuming points1, points2, and intrinsics are given
# E_ransac = calculate_essential_matrix_ransac(normalized_points1, normalized_points2)
# print("\nEssential Matrix(RANSAC):")
# print(E_ransac)


# R, t = decompose_essential_matrix(E_ransac)
# print("\nR:")
# print(R)
# print("\nt:")
# print(t)

# R, t = decompose_essential_matrix(E)
# print("\nR:")
# print(R)
# print("\nt:")
# print(t)


# R, t = decompose_essential_matrix(E_ransac)
# print("\nR:")
# print(R)
# print("\nt:")
# print(t)

correspondences_folder = '/Users/hewanshrestha/Desktop/3dcv_project/stage1/box/correspondences/'
intrinsics_file = '/Users/hewanshrestha/Desktop/3dcv_project/stage1/box/gt_camera_parameters.json'

with open(intrinsics_file, 'r') as f:
    intrinsics_data = json.load(f)
K = np.array(intrinsics_data['intrinsics'])

# Get a list of all correspondence files
correspondence_files = [f for f in os.listdir(correspondences_folder) if f.endswith('.txt')]

positive_depth_count = 0
negative_depth_count = 0

for correspondences_file in correspondence_files:
    points1, points2 = get_correspondence_from_file(os.path.join(correspondences_folder, correspondences_file))

    # Normalize the points
    normalized_points1 = camera_points(points1, K)
    normalized_points2 = camera_points(points2, K)

    # E_cv, mask = cv2.findEssentialMat(normalized_points1, normalized_points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    # print(f"Essential Matrix(cv2):")
    # print(E_cv)

    # E = calculate_essential_matrix(normalized_points1, normalized_points2)
    # print(f"\nEssential Matrix:")
    # print(E)

    # Assuming points1, points2, and intrinsics are given
    E_ransac = calculate_essential_matrix_ransac(normalized_points1, normalized_points2)
    # print("\nEssential Matrix(RANSAC):")
    # print(E_ransac)

    R, t = decompose_essential_matrix(E_ransac)
    # print("\nR:")
    # print(R)
    # print("\nt:")
    # print(t)

    # R, t = decompose_essential_matrix(E)
    # print("\nR:")
    # print(R)
    # print("\nt:")
    # print(t)

    # R, t = decompose_essential_matrix(E_ransac)
    # print("\nR:")
    # print(R)
    # print("\nt:")
    # print(t)

    visualize_results(R, t)

# Print the total counts
print("Total number of points with positive depth: ", positive_depth_count)
print("Total number of points with negative depth: ", negative_depth_count)

