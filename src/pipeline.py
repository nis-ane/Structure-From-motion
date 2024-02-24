"""
This is the main script where all integration of whole pipeline is to be done
"""

import argparse
import os
import json
from src.utils import *
from src.triangulation import triangulate_pts
from src.pose_estimation import (
    estimate_pose_Linear_PnP_RANSAC,
    estimate_pose_Linear_PnP,
)
from src.correspondence import get_3d_to_2d_correspondence
from src.visualize import *
import cv2


def run_pipeline(cam_parameters, image_folder, correspondence_folder=None):
    # Iterate over the images
    # Compute the Rotation and translation between two images
    # Compute 3d points using triangulation
    # Carry out bundle adjustment
    # Add extra image
    # Compute rotation and translation with image with great correspondence
    # Compute 3d point of new points
    # Carry out bundle adjustment.

    K = np.array(cam_parameters["intrinsics"])
    is_first = True
    prev_image_idx = None
    x1_prev = None
    X = None
    poses = []

    for frame_n, image_name in enumerate(sorted(os.listdir(image_folder))):
        image_idx = int(image_name.split(".")[0])
        if is_first:
            prev_image_idx = image_idx
            is_first = False
            RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            P_prev = np.dot(K, RT1)
            poses.append(RT1)
        else:
            curr_image_idx = image_idx
            correspondence_file_name = f"{prev_image_idx}_{curr_image_idx}.txt"
            correspondence_file_path = os.path.join(
                correspondence_folder, correspondence_file_name
            )
            x1_curr, x2_curr = get_correspondence_from_file(correspondence_file_path)
            img2 = cv2.imread(os.path.join(image_folder, image_name))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            colors = [img2[int(v)][int(u)] for (u, v, _) in x2_curr]

            if X is None:
                RT_curr = np.array(cam_parameters["extrinsics"][image_name])[
                    :3, :
                ]  # This will be replaced by below line.
                # RT_curr = estimate_pose_Essential_mat()
                P_curr = np.dot(K, RT_curr)
                poses.append(RT_curr)
                X = triangulate_pts(x1_curr, x2_curr, P_prev, P_curr)

            else:
                X_f, x2_f = get_3d_to_2d_correspondence(X, x1_prev, x1_curr, x2_curr)
                assert len(X_f) > 40
                R, T, _ = estimate_pose_Linear_PnP_RANSAC(x2_f, X_f, K)
                RT_curr = np.hstack((R, T))
                P_curr = np.dot(K, RT_curr)
                poses.append(RT_curr)
                X = triangulate_pts(x1_curr, x2_curr, P_prev, P_curr)
                # OPtimize X R and T using bundle adjustment

            visualise_poses_and_3d_points_with_gt(
                poses, X[:, :3], cam_parameters, n=frame_n + 1, colors=colors
            )
            # visualise_poses_with_gt(
            #         poses, cam_parameters, n=frame_n+2
            #     )  # to visualize without 3d points for computational efficiency.

            # visualise_gt_poses(cam_parameters) # To visulaize all gt poses
            x1_prev = x2_curr
            P_prev = P_curr
            prev_image_idx = curr_image_idx
            if frame_n > 5:
                break


if __name__ == "__main__":
    print("Estimating camera extrinsic parameters")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="box",
        help="Name of dataset to generate the parameters. For stage 1 possible values are 'box' and 'boot'",
    )
    parser.add_argument(
        "-s",
        "--stage",
        type=int,
        default=1,
        help="Stage of Project. The resources precomputed assumption is based on the stage of project",
    )
    args = parser.parse_args()

    root_folder = f"./Stage_{args.stage}/stage1"
    dataset_folder = os.path.join(root_folder, args.dataset)
    assert os.path.exists(
        dataset_folder
    ), "Dataset does not exist. Check the folder of dataset is inside the folder submission"

    image_folder = os.path.join(dataset_folder, "images")
    assert os.path.exists(
        image_folder
    ), "Image Folder missing inside the dataset folder"

    pose_json_path = os.path.join(dataset_folder, "gt_camera_parameters.json")
    with open(pose_json_path) as f:
        cam_parameters = json.load(f)

    correspondence_folder = os.path.join(dataset_folder, "correspondences")

    # visualise_gt_poses(cam_parameters)
    if args.stage == 1:
        run_pipeline(cam_parameters, image_folder, correspondence_folder)
    elif args.stage == 2:
        run_pipeline(cam_parameters, image_folder)
