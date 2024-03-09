import numpy as np
from src.pose_estimation import estimate_pose_Essential_Matrix
from src.frame import Frame
import cv2
import os
import json


def test_pose_from_essential_mat():
    image_folder = "./Stage_1/stage1/box/images"
    pose_json_path = "./Stage_1/stage1/box/gt_camera_parameters.json"
    with open(pose_json_path) as f:
        cam_parameters = json.load(f)
    K = np.array(cam_parameters["intrinsics"])

    image_1 = cv2.imread(os.path.join(image_folder, "00000.jpg"))
    image_2 = cv2.imread(os.path.join(image_folder, "00002.jpg"))
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    image_idx_1 = 0
    image_idx_2 = 2
    frame_1 = Frame(image_1, image_idx_1, K)
    frame_2 = Frame(image_2, image_idx_2, K)
    R, T = estimate_pose_Essential_Matrix(frame_1, frame_2)
    frame_2.R = R
    frame_2.T = T
    frame_2.compute_projection_matrix()

    eps = 1e-6
    r1 = np.array(cam_parameters["extrinsics"]["00002.jpg"])[:3, :3]
    r2 = frame_2.R
    rot_error = (np.trace(r1 @ r2.T) - 1) / 2
    rot_error = np.clip(rot_error, -1.0 + eps, 1.0 - eps)
    rot_error = np.arccos(rot_error)
    assert abs(rot_error) < 0.05
