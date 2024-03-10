import numpy as np
from src.utils import get_correspondence_from_file
import json
import cv2
from scipy.linalg import svd
from src.triangulation import triangulate_pts
from src.visualize import visualise_pose_and_3d_points
import random
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
        x, y, _ = points1[i]
        xp, yp, _ = points2[i]
        A[i] = [xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, 1]

    # Solve the equation Af = 0 using SVD
    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)

    # Enforce the rank-2 constraint on E
    U, _, Vt = np.linalg.svd(E)
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), Vt))

    return E


def ransac_essential_matrix(points1, points2, T=10, threshold=1e-3, k_max=1000):
    random.seed(0)
    best_E = None
    best_inliers = []

    # Convert points to homogeneous coordinates
    points1 = np.concatenate([points1, np.ones((points1.shape[0], 1))], axis=1)
    points2 = np.concatenate([points2, np.ones((points2.shape[0], 1))], axis=1)

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
            updated_E = calculate_essential_matrix([points1[i] for i in all_inliers], [points2[i] for i in all_inliers])
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

    return best_E


def decompose_essential_matrix(E):
    # Compute SVD of E
    U, _, Vt = np.linalg.svd(E)

    # Ensure it's a right-handed coordinate system
    if np.linalg.det(np.dot(U, Vt)) < 0:
        Vt = -Vt

    # Compute rotation matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = np.dot(np.dot(U, W), Vt)
    R2 = np.dot(np.dot(U, W.T), Vt)

    # If the determinant of R is -1, flip the sign of the rotation matrix
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

    return [(c1, r1), (c2, r2), (c3, r3), (c4, r4)]



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






correspondences_file = '/Users/hewanshrestha/Desktop/3dcv_project/stage1/box/correspondences/68_75.txt'
intrinsics_file = '/Users/hewanshrestha/Desktop/3dcv_project/stage1/box/gt_camera_parameters.json'

points1, points2 = get_correspondence_from_file(correspondences_file)
with open(intrinsics_file, 'r') as f:
    intrinsics_data = json.load(f)
K = np.array(intrinsics_data['intrinsics'])

# print(f"Calibration Matrix(K): {K}")
print(f"K: {K.shape}")

# print(f"Points1: {points1}")
print(f"points1: {points1.shape}")

# print(f"Points2: {points2}")
print(f"points2: {points2.shape}")

# Normalize the points
normalized_points1 = camera_points(points1, K)
# print(f"Normalized points1: {normalized_points1}")
print(f"Normalized points1: {normalized_points1.shape}")

normalized_points2 = camera_points(points2, K)
# print(f"Normalized points2: {normalized_points2}")
print(f"Normalized points2: {normalized_points2.shape}")

E_cv, mask = cv2.findEssentialMat(normalized_points1, normalized_points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
print(f"Essential Matrix(cv2):")
print(E_cv)

# Assuming points1, points2, and intrinsics are given
E_ransac = ransac_essential_matrix(normalized_points1, normalized_points2)
print("\nEssential Matrix(RANSAC):")
print(E_ransac)


# Convert points to homogeneous coordinates
normalized_points1_3d = np.column_stack([normalized_points1, np.ones(normalized_points1.shape[0])])
normalized_points2_3d = np.column_stack([normalized_points2, np.ones(normalized_points2.shape[0])])


def is_in_front_of_camera(point, R, t):
    # Apply the camera transformation
    transformed_point = np.dot(R, point - t)
    # A point is in front of the camera if its z-coordinate is positive
    return transformed_point[2] > 0

def count_points_in_front_of_both_cameras(first_points, second_points, R, t):
    return np.sum([is_in_front_of_camera(point, R, t) for point in first_points]) + \
           np.sum([is_in_front_of_camera(point, R, t) for point in second_points])

poses = decompose_essential_matrix(E_ransac)
counts = [count_points_in_front_of_both_cameras(normalized_points1_3d, normalized_points2_3d, R, t) for t, R in poses]
best_pose_index = np.argmax(counts)
best_pose = poses[best_pose_index]

print(f"Best pose: {best_pose_index + 1}")

best_t, best_R = best_pose
print(f"Best R: {best_R}")
print(f"Best t: {best_t}")

visualize_results(best_R, best_t)



# correspondences_folder = '/Users/hewanshrestha/Desktop/3dcv_project/stage1/box/correspondences/'
# count_missing_E = 0
# # Loop over all files in the correspondences folder
# for filename in os.listdir(correspondences_folder):
#     if filename.endswith('.txt'):
#         correspondences_file = os.path.join(correspondences_folder, filename)
#         points1, points2 = get_correspondence_from_file(correspondences_file)

#         # Normalize the points
#         normalized_points1 = camera_points(points1, K)
#         normalized_points2 = camera_points(points2, K)

#         E_ransac = ransac_essential_matrix(normalized_points1, normalized_points2)

#         if E_ransac is None:
#             print(f"Failed to find essential matrix for {filename}")
#             count_missing_E += 1
#             continue

#         # Convert points to homogeneous coordinates
#         normalized_points1_3d = np.column_stack([normalized_points1, np.ones(normalized_points1.shape[0])])
#         normalized_points2_3d = np.column_stack([normalized_points2, np.ones(normalized_points2.shape[0])])

#         poses = decompose_essential_matrix(E_ransac)
#         counts = [count_points_in_front_of_both_cameras(normalized_points1_3d, normalized_points2_3d, R, t) for t, R in poses]
#         best_pose_index = np.argmax(counts)
#         best_pose = poses[best_pose_index]

#         best_t, best_R = best_pose

#         # Create the projection matrices
#         P_R_t = np.vstack((np.hstack((best_R, best_t)), [0, 0, 0, 1]))
#         I_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

#         # Calculate projection matrices
#         P1 = np.dot(K, I_0)
#         P2 = P_R_t

#         # Remove the last row of zeros
#         P1 = P1[:3, :]
#         P2 = P2[:3, :]

#         RT1 = np.eye(4)[:3, :]
#         RT2 = P_R_t

#         # Triangulate points
#         X = triangulate_pts(points1, points2, P1, P2)
#         X = X[:, :3] / X[:, 3:]

#         print(f"X for {filename}: ", X[2, :])


# # Print the number of files with None essential matrix
# print(f"Number of files with None essential matrix: {count_missing_E}")