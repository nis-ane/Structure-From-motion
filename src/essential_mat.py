"""
This module contains code for
1. Estimation of essential matrix
2. Decompositon of Essential Matrix to R and T
3. Resolve the ambiguity of R and T decomposed from Essential Matrix
"""

import os
import cv2
import numpy as np
import json
from src.utils import get_correspondence_from_file
from src.triangulation import triangulate_pts
from src.visualize import visualise_pose_and_3d_points

class EssentialMatrixEstimation:
    def __init__(self, correspondences_file, intrinsics_file):
        # Load correspondences and intrinsics
        self.points1, self.points2 = get_correspondence_from_file(correspondences_file)
        with open(intrinsics_file, 'r') as f:
            intrinsics_data = json.load(f)
        self.K = np.array(intrinsics_data['intrinsics'])

    def normalize_points(self, points, intrinsics):
        # Normalize 2D points
        points = points[:, :2]
        points_homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
        normalized_points_homogeneous = np.dot(points_homogeneous, np.linalg.inv(intrinsics).T)
        normalized_points = normalized_points_homogeneous[:, :2] / normalized_points_homogeneous[:, 2, np.newaxis]
        return normalized_points
    
    def calculate_essential_matrix(self, points1, points2):

        # points1 = self.normalize_points(points1, K)
        # points2 = self.normalize_points(points2, K)

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


    def calculate_essential_matrix_ransac(self, points1, points2, iterations=1000, threshold=1e-3):
        np.random.seed(0)
        best_inliers = 0
        best_E = None

        for _ in range(iterations):
            # Randomly select 8 point correspondences
            indices = np.random.choice(len(points1), 8, replace=False)
            points1_sample = points1[indices]
            points2_sample = points2[indices]

            # Calculate the essential matrix from the sample
            E_sample = self.calculate_essential_matrix(points1_sample, points2_sample)

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

    def calculate_normalized_essential_matrix(self, points1, points2, K):
        # Normalize the points
        points1_normalized = self.normalize_points(points1, K)
        points2_normalized = self.normalize_points(points2, K)

        # Convert to homogeneous coordinates
        points1_normalized = np.hstack((points1_normalized, np.ones((points1_normalized.shape[0], 1))))
        points2_normalized = np.hstack((points2_normalized, np.ones((points2_normalized.shape[0], 1))))

        # Calculate the transformation matrices
        T1 = self.calculate_transformation_matrix(points1_normalized)
        T2 = self.calculate_transformation_matrix(points2_normalized)

        # Transform the points
        points1_transformed = np.dot(T1, points1_normalized.T).T
        points2_transformed = np.dot(T2, points2_normalized.T).T

        # Calculate the essential matrix from the transformed points
        E_transformed = self.calculate_essential_matrix_ransac(points1_transformed, points2_transformed)

        # Denormalize the essential matrix
        E = np.dot(T1.T, np.dot(E_transformed, T2))

        return E

        
    def calculate_transformation_matrix(self, points):
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

    # Decompose the essential matrix
    def decompose_essential_matrix(self, E):
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

        R, t = self.enforce_positive_depth(R, t, self.points1, self.points2)

        return R, t

    def enforce_positive_depth(self, R, t, points1, points2):
        # Calculate the camera centers
        C1 = np.zeros((3, 1))
        C2 = -np.dot(R.T, t)

        # Calculate the 3D points
        X = triangulate_pts(points1, points2, np.eye(3, 4), np.hstack((R, t)))

        X = X[:, :3] / X[:, 3:]

        # Calculate the depth of the 3D points
        C1 = C1.reshape(1, 3)
        depth1 = np.dot(R[2], (X - C1).T)

        C2 = C2.reshape(1, 3)
        depth2 = np.dot(R[2], (X - C2).T)
        

        # If the depth is negative, flip the sign of the translation vector
        if np.sum(depth1 < 0) > np.sum(depth2 < 0):
            t = -t

        return R, t
    
    def visualize_results(self, R, t):
        # Create the projection matrices
        P_R_t = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

        I_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        # Calculate projection matrices
        P1 = np.dot(self.K, I_0)
        P2 = P_R_t

        # Remove the last row of zeros
        P1 = P1[:3, :]
        P2 = P2[:3, :]

        RT1 = np.eye(4)[:3, :]
        RT2 = P_R_t

        # Read the second image
        # img2 = cv2.imread(image_path)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Triangulate points
        X = triangulate_pts(self.points1, self.points2, P1, P2)

        print("X:", X[2, :])

        # Extract colors
        # colors = [img2[int(v)][int(u)] for (u, v, w) in self.points2]

        # Visualize 3D points and camera poses
        visualise_pose_and_3d_points([RT1, RT2], X[:, :3])

    def run(self):
        # Normalize 2D points
        # points1_normalized = self.normalize_points(self.points1, self.K)
        # points2_normalized = self.normalize_points(self.points2, self.K)

        # Calculate essential matrix
        E = self.calculate_normalized_essential_matrix(self.points1, self.points2, self.K)

        # Decompose the essential matrix
        R, t = self.decompose_essential_matrix(E)

        # Visualize the results
        self.visualize_results(R, t)

def main():
    correspondences_file = '/Users/hewanshrestha/Desktop/3dcv_project/Stage_1/submission/box/correspondences/53_88.txt'
    intrinsics_file = '/Users/hewanshrestha/Desktop/3dcv_project/Stage_1/submission/box/gt_camera_parameters.json'
    
    essential_matrix_estimator = EssentialMatrixEstimation(correspondences_file, intrinsics_file)
    # Run the estimation and decomposition
    essential_matrix_estimator.run()

if __name__ == "__main__":
    main()
