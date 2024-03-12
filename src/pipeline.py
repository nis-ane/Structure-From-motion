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
    get_3d_to_2d_correspondence,
    get_2d_to_2d_correspondence,
    associate_correspondences,
)
from src.map import Map, register_frames_with_map
from src.frame import Frame
from src.optimize import optimize_pose_and_map
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
    map_ = Map()  # This saves the whole 3d point cloud
    is_first = True
    frames = []  # This saves the frames with its poses
    if correspondence_folder is not None:
        correspondence = True
    else:
        correspondence = False

    for frame_n, image_name in enumerate(sorted(os.listdir(image_folder))):
        image_idx = int(image_name.split(".")[0])
        image = cv2.imread(os.path.join(image_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initilization of first frame(Also the reference frame). This camera is to be in the origin.
        if is_first:
            frame_prev = Frame(image, image_idx, K, correspondence=correspondence)
            # RT_curr = np.array(cam_parameters["extrinsics"][image_name])[:3, :]
            # R = RT_curr[:3, :3]
            # T = RT_curr[:3, 3:4]
            # frame_prev.T = T  # Initialize the translation to be zero. The camera center is in origin
            # frame_prev.R = R
            frame_prev.T = np.zeros(
                (3, 1)
            )  # Initialize the translation to be zero. The camera center is in origin
            frame_prev.R = np.eye(
                3
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
                x_prev, x_curr = get_correspondence_from_file(correspondence_file_path)
                frame_prev.update_keypoints_using_correspondence(x_prev)
                frame_curr.update_keypoints_using_correspondence(x_curr)

            else:  # Stage 2 when correspondece is to be estimated
                x_prev, x_curr = get_2d_to_2d_correspondence(
                    frame_prev, frame_curr
                )  # This should specify
                # frame_prev.update_keypoints_using_correspondence(x_prev)
                # frame_curr.update_keypoints_using_correspondence(x_curr)

            associate_correspondences(
                frame_prev, frame_curr
            )  # We need to find the matches between map and curr frame and update the keypoints of prev frame

            if (
                len(map_.X) == 0 or len(frame_curr.intersect_idx) < 6
            ):  # Atleast 6 correspondence is required for Linear PnP algorithm
                # RT_curr = np.array(cam_parameters["extrinsics"][image_name])[:3, :]
                # R = RT_curr[:3, :3]
                # T = RT_curr[:3, 3:4]
                R, T = estimate_pose_Essential_Matrix(
                    frame_prev, frame_curr
                )  # When there is no map or match is less than 6 we go to 2d-2d estimation.
                # frame_curr.R = np.dot(frame_prev.R, R)                          # Since 2D-2D gives relative rotation and Translation with respect to first frame.
                # frame_curr.T = np.add(frame_prev.T, T)

                frame_curr.R = (
                    R  # 3D-2D correspondence gives absolute Rotation and Translation.
                )
                frame_curr.T = T
                print(frame_curr.R, frame_curr.T)

                frame_curr.compute_projection_matrix()
                # print("RT:",frame_curr.RT, frame_prev.RT)
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
                # RT_curr = np.array(cam_parameters["extrinsics"][image_name])[:3, :]
                # R = RT_curr[:3, :3]
                # T = RT_curr[:3, 3:4]
                frame_curr.R = (
                    R  # 3D-2D correspondence gives absolute Rotation and Translation.
                )
                frame_curr.T = T
                # print(frame_curr.R, frame_curr.T)
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

            optimize_pose_and_map(
                frames, map_, frame_n - 1
            )  # Use bundle Adjustment to optimize the frames and map.
            frame_prev = frames[-1]  # Update prev frame for next Iteration
            visualise_poses_with_gt(
                [frame.RT for frame in frames], cam_parameters, n=2
            )  #
            visualise_poses_and_3d_points_with_gt(
                [frame.RT for frame in frames],
                map_.X[:, :3],
                cam_parameters,
                n=2,
                colors=map_.color,
            )
            # print("Number of frames:",len(frames))
            # for frame in frames:
            #     print(frame.RT)
            # visualise_poses_with_gt(
            #         [frame.RT for frame in frames], cam_parameters, n=2
            #     )  # to visualize without 3d points for computational efficiency.
            break
            # visualise_gt_poses(cam_parameters)
            if frame_n > 3:
                break


# def run_pipeline(cam_parameters, image_folder, correspondence_folder=None):
#     # Iterate over the images
#     # Compute the Rotation and translation between two images
#     # Compute 3d points using triangulation
#     # Carry out bundle adjustment
#     # Add extra image
#     # Compute rotation and translation with image with great correspondence
#     # Compute 3d point of new points
#     # Carry out bundle adjustment.

#     K = np.array(cam_parameters["intrinsics"])
#     is_first = True
#     prev_image_idx = None
#     x1_prev = None
#     X = None
#     poses = []

#     for frame_n, image_name in enumerate(sorted(os.listdir(image_folder))):
#         image_idx = int(image_name.split(".")[0])
#         if is_first:
#             prev_image_idx = image_idx
#             is_first = False
#             RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
#             P_prev = np.dot(K, RT1)
#             poses.append(RT1)
#         else:
#             curr_image_idx = image_idx
#             correspondence_file_name = f"{prev_image_idx}_{curr_image_idx}.txt"
#             correspondence_file_path = os.path.join(
#                 correspondence_folder, correspondence_file_name
#             )
#             x1_curr, x2_curr = get_correspondence_from_file(correspondence_file_path)
#             img2 = cv2.imread(os.path.join(image_folder, image_name))
#             img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#             colors = [img2[int(v)][int(u)] for (u, v, _) in x2_curr]

#             if X is None:
#                 RT_curr = np.array(cam_parameters["extrinsics"][image_name])[
#                     :3, :
#                 ]  # This will be replaced by below line.
#                 # RT_curr = estimate_pose_Essential_mat()
#                 P_curr = np.dot(K, RT_curr)
#                 poses.append(RT_curr)
#                 X = triangulate_pts(x1_curr, x2_curr, P_prev, P_curr)

#             else:
#                 X_f, x2_f = get_3d_to_2d_correspondence(X, x1_prev, x1_curr, x2_curr)
#                 assert len(X_f) > 40
#                 R, T, _ = estimate_pose_Linear_PnP_RANSAC(x2_f, X_f, K)
#                 RT_curr = np.hstack((R, T))
#                 P_curr = np.dot(K, RT_curr)
#                 poses.append(RT_curr)
#                 X = triangulate_pts(x1_curr, x2_curr, P_prev, P_curr)
#                 # OPtimize X R and T using bundle adjustment

#             visualise_poses_and_3d_points_with_gt(
#                 poses, X[:, :3], cam_parameters, n=frame_n + 1, colors=colors
#             )
#             # visualise_poses_with_gt(
#             #         poses, cam_parameters, n=frame_n+2
#             #     )  # to visualize without 3d points for computational efficiency.

#             # visualise_gt_poses(cam_parameters) # To visulaize all gt poses
#             x1_prev = x2_curr
#             P_prev = P_curr
#             prev_image_idx = curr_image_idx
#             if frame_n > 5:
#                 break


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

    root_folder = f"./Stage_1/stage{args.stage}"
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
