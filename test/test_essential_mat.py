import numpy as np
from src.pose_estimation import estimate_pose_Essential_Matrix
from src.utils import get_correspondence_from_file
from src.essential_mat import calculate_essential_matrix, camera_points
from src.frame import Frame
import cv2
import os
import json


image_folder = "./data/stage1/box/images"
pose_json_path = "./data/stage1/box/gt_camera_parameters.json"
correspondence_folder = "./data/stage1/box/correspondences"
with open(pose_json_path) as f:
    cam_parameters = json.load(f)
K = np.array(cam_parameters["intrinsics"])

image_1 = cv2.imread(os.path.join(image_folder, "00000.jpg"))
image_2 = cv2.imread(os.path.join(image_folder, "00001.jpg"))
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
image_idx_1 = 0
image_idx_2 = 1
frame_1 = Frame(image_1, image_idx_1, K)
frame_2 = Frame(image_2, image_idx_2, K)
correspondence_file_name = f"{frame_1.img_id}_{frame_2.img_id}.txt"
correspondence_file_path = os.path.join(correspondence_folder, correspondence_file_name)
x_prev, x_curr = get_correspondence_from_file(correspondence_file_path)
frame_1.update_keypoints_using_correspondence(x_prev)
frame_2.update_keypoints_using_correspondence(x_curr)


def test_pose_from_essential_mat():
    R, T = estimate_pose_Essential_Matrix(frame_1, frame_2)
    print("R:", R)
    frame_2.R = R
    frame_2.T = T
    frame_2.compute_projection_matrix()

    eps = 1e-6
    r1 = np.array(cam_parameters["extrinsics"]["00001.jpg"])[:3, :3]
    r2 = frame_2.R
    print(r1)
    print(r2)
    rot_error = (np.trace(r1 @ r2.T) - 1) / 2
    rot_error = np.clip(rot_error, -1.0 + eps, 1.0 - eps)
    rot_error = np.arccos(rot_error)
    assert abs(rot_error) < 0.5


def test_rank_essential_matrix():
    points_1 = frame_1.keypoints[frame_1.matched_idx]
    points_2 = frame_2.keypoints[frame_2.matched_idx]
    normalised_points_1 = camera_points(points_1, K)
    normalised_points_2 = camera_points(points_2, K)
    E = calculate_essential_matrix(normalised_points_1, normalised_points_2)
    assert np.linalg.matrix_rank(E) == 2


def test_norm_essential_matrix():
    points_1 = frame_1.keypoints[frame_1.matched_idx]
    points_2 = frame_2.keypoints[frame_2.matched_idx]
    normalised_points_1 = camera_points(points_1, K)
    normalised_points_2 = camera_points(points_2, K)
    E = calculate_essential_matrix(normalised_points_1, normalised_points_2)
    assert np.round(np.linalg.norm(E)) == 1


# def test_compare_cv2_essential_mat():
#     points_1 = frame_1.keypoints[frame_1.matched_idx]
#     points_2 = frame_2.keypoints[frame_2.matched_idx]
#     normalised_points_1 = camera_points(points_1,K)
#     normalised_points_2 = camera_points(points_2,K)
#     E = calculate_essential_matrix(normalised_points_1, normalised_points_2)
#     E_cv, mask = cv2.findEssentialMat(points_1[:,:2], points_2[:,:2], K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#     print(E)
#     print(E_cv)
#     print(np.count_nonzero(mask))
#     eps = 1e-6
#     error = (np.trace(E_cv @ E.T) - 1) / 2
#     error = np.clip(error, -1.0 + eps, 1.0 - eps)
#     error = np.arccos(error)
#     assert abs(error) < 0.5
