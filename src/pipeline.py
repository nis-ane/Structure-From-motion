"""
This is the main script where all integration of whole pipeline is to be done
"""

import argparse
import os
import json
from src.utils import *
from src.triangulation import triangulate_pts
from src.pose_estimation import (
    estimate_pose_3d_2d_mapping,
    estimate_pose_Essential_Matrix,
)
from src.correspondence import (
    get_2d_to_2d_correspondence,
    associate_correspondences,
)
from src.map import Map, register_frames_with_map
from src.frame import Frame
from src.optimize import optimize_pose_and_map
from src.visualize import *
import cv2


def run_pipeline(
    cam_parameters, image_folder, dataset_folder, correspondence_folder=None
):
    # Iterate over the images
    # Compute the Rotation and translation between two images
    # Compute 3d points using triangulation
    # Carry out bundle adjustment
    # Add extra image
    # Compute rotation and translation with image with great correspondence
    # Compute 3d point of new points
    # Carry out bundle adjustment.

    K = np.array(cam_parameters["intrinsics"])
    map_ = Map()  # This saves the whole 3d point cloud
    is_first = True
    frames = []  # This saves the frames with its poses
    if correspondence_folder is not None:
        correspondence = True
    else:
        correspondence = False

    for frame_n, image_name in enumerate(sorted(os.listdir(image_folder))):
        print(image_name)
        image_idx = int(image_name.split(".")[0])
        image = cv2.imread(os.path.join(image_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initilization of first frame(Also the reference frame). This camera is to be in the origin.
        if is_first:
            frame_prev = Frame(image, image_idx, K, correspondence=correspondence)
            R_error_matrix = np.random.rand(3, 3) * 1e-8
            T_error_matrix = np.random.rand(3, 1) * 1e-8
            frame_prev.T = (
                np.zeros((3, 1)) + T_error_matrix
            )  # Initialize the translation to be zero. The camera center is in origin
            frame_prev.R = (
                np.eye(3) + R_error_matrix
            )  # Initilize the rotation to be identity. Aligned with the axes.
            frame_prev.compute_projection_matrix()  # Computes projection matrix with above R and T
            frames.append(frame_prev)
            is_first = False

        else:
            frame_curr = Frame(image, image_idx, K, correspondence=correspondence)

            if correspondence_folder:  # Stage 1 when correspondence is provided
                correspondence_file_name = (
                    f"{frame_prev.img_id}_{frame_curr.img_id}.txt"
                )

                correspondence_file_path = os.path.join(
                    correspondence_folder, correspondence_file_name
                )

                updated_prev_idx = -2
                while not os.path.exists(correspondence_file_path):
                    frame_prev = frames[updated_prev_idx]
                    correspondence_file_name = (
                        f"{frame_prev.img_id}_{frame_curr.img_id}.txt"
                    )
                    correspondence_file_path = os.path.join(
                        correspondence_folder, correspondence_file_name
                    )
                    updated_prev_idx -= 1
                print(correspondence_file_name)

                x_prev, x_curr = get_correspondence_from_file(correspondence_file_path)
                frame_prev.update_keypoints_using_correspondence(x_prev)
                frame_curr.update_keypoints_using_correspondence(x_curr)

            else:  # Stage 2 when correspondece is to be estimated
                x_prev, x_curr = get_2d_to_2d_correspondence(frame_prev, frame_curr)

            associate_correspondences(
                frame_prev, frame_curr
            )  # We need to find the matches between map and curr frame and update the keypoints of prev frame

            # Atleast 6 correspondence is required for Linear PnP algorithm
            if len(map_.X) == 0 or len(frame_curr.intersect_idx) < 6:

                R, T = estimate_pose_Essential_Matrix(frame_prev, frame_curr)
                frame_curr.R = R
                frame_curr.T = T

                frame_curr.compute_projection_matrix()
                X = triangulate_pts(
                    x_prev[frame_prev.disjoint_idx],
                    x_curr[frame_curr.disjoint_idx],
                    frame_prev.P,
                    frame_curr.P,
                )
                colors = [
                    frame_curr.image[int(v)][int(u)]
                    for (u, v, _) in x_curr[frame_curr.disjoint_idx]
                ]

                register_frames_with_map(frame_prev, frame_curr, map_, X)
                map_.update_map(X, colors)

                frames.append(frame_curr)

            else:
                R, T = estimate_pose_3d_2d_mapping(
                    map_, frame_curr
                )  # Estimates R and T using 3D-2D correspondences.

                frame_curr.R = R
                frame_curr.T = T
                frame_curr.compute_projection_matrix()
                X = triangulate_pts(
                    x_prev[frame_prev.disjoint_idx],
                    x_curr[frame_curr.disjoint_idx],
                    frame_prev.P,
                    frame_curr.P,
                )
                colors = [
                    frame_curr.image[int(v)][int(u)]
                    for (u, v, _) in x_curr[frame_curr.disjoint_idx]
                ]

                register_frames_with_map(frame_prev, frame_curr, map_, X)
                map_.update_map(X, colors)
                frames.append(frame_curr)

            try:  # There is some issue of Gauze freedom which leads to non convergence
                optimize_pose_and_map(
                    frames, map_
                )  # Use bundle Adjustment to optimize the frames and map.
            except:
                pass
            frame_prev = frames[-1]  # Update prev frame for next Iteration

    # Save the Pose and 3d point cloud
    point_cloud = trimesh.PointCloud(vertices=map_.X[:, :3], colors=map_.color)
    point_cloud.export(os.path.join(dataset_folder, "estimated_points.ply"))
    pose_out_path = os.path.join(dataset_folder, "estimated_camera_parameters.json")
    estimated_pose = {}
    frame_estimated_pose = {}
    for frame_n, image_name in enumerate(sorted(os.listdir(image_folder))):
        frame_estimated_pose[image_name] = frames[frame_n].RT.tolist()

    estimated_pose["extrinsics"] = frame_estimated_pose

    with open(pose_out_path, "w") as fp:
        json.dump(estimated_pose, fp)


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
    parser.add_argument(
        "-t",
        "--gt",
        type=int,
        default=1,
        help="Whether gt is available or not",
    )
    args = parser.parse_args()

    root_folder = f"./data/stage{args.stage}"
    dataset_folder = os.path.join(root_folder, args.dataset)
    assert os.path.exists(
        dataset_folder
    ), "Dataset does not exist. Check the folder of dataset is inside the folder submission"

    image_folder = os.path.join(dataset_folder, "images")
    assert os.path.exists(
        image_folder
    ), "Image Folder missing inside the dataset folder"

    if args.gt == 1:
        pose_json_path = os.path.join(dataset_folder, "gt_camera_parameters.json")
    elif args.stage == 1:
        pose_json_path = os.path.join(dataset_folder, "camera_parameters.json")
    else:
        pose_json_path = os.path.join(dataset_folder, "poses.json")
    with open(pose_json_path) as f:
        cam_parameters = json.load(f)

    correspondence_folder = os.path.join(dataset_folder, "correspondences")

    if args.stage == 1:
        run_pipeline(
            cam_parameters, image_folder, dataset_folder, correspondence_folder
        )
    elif args.stage == 2:
        run_pipeline(cam_parameters, image_folder, dataset_folder)
