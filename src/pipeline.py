"""
This is the main script where all integration of whole pipeline is to be done
"""

import argparse
import os
import json
from src.utils import *
from src.triangulation import triangulate_pts
from src.visualize import *
import cv2


def run_pipeline(K, image_folder, correspondence_folder=None):
    # Iterate over the images
    # Compute the Rotation and translation between two images
    # Compute 3d points using triangulation
    # Carry out bundle adjustment
    # Add extra image
    # Compute rotation and translation with image with great correspondence
    # Compute 3d point of new points
    # Carry out bundle adjustment.

    K = np.array(cam_paramters["intrinsics"])
    is_first = True
    prev_image_idx = None
    for image_name in sorted(os.listdir(image_folder)):
        image_idx = int(image_name.split(".")[0])
        if is_first:
            prev_image_idx = image_idx
            is_first = False
            RT1 = np.array(cam_paramters["extrinsics"][image_name])[:3, :]
            P1 = np.dot(K, RT1)
        else:
            curr_image_idx = image_idx
            RT2 = np.array(cam_paramters["extrinsics"][image_name])[:3, :]
            P2 = np.dot(K, RT2)
            correspondence_file_name = f"{prev_image_idx}_{curr_image_idx}.txt"
            correspondence_file_path = os.path.join(
                correspondence_folder, correspondence_file_name
            )
            x1, x2 = get_correspondence_from_file(correspondence_file_path)
            img2 = cv2.imread(os.path.join(image_folder, image_name))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            colors = [img2[int(v)][int(u)] for (u, v, w) in x2]
            X = triangulate_pts(x1, x2, P1, P2)
            visualise_pose_and_3d_points([RT1, RT2], X[:, :3], colors)
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

    root_folder = f"./Stage_{args.stage}/submission"
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
        cam_paramters = json.load(f)

    correspondence_folder = os.path.join(dataset_folder, "correspondences")

    if args.stage == 1:
        run_pipeline(cam_paramters, image_folder, correspondence_folder)
    elif args.stage == 2:
        run_pipeline(cam_paramters, image_folder)
