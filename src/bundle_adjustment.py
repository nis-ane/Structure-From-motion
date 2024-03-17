from src.utils import project_3D_to_2D
from src.visualize import visualize_reprojection_error
import numpy as np
from src.sparse_ba import SBA

# from src.jacobian import compute_A_B
from src.jacobian import compute_pose_jacobian_mat, compute_X_jacobian_mat
from src.utils import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix


def compute_reprojection_error_total(frames, map_):
    total_error = 0
    for frame in frames:
        X = map_.X[frame.index_kp_3d]
        x_proj = project_3D_to_2D(X, frame.P)
        x = frame.keypoints[frame.triangulated_idx]
        frame_error = np.linalg.norm(x - x_proj)
        assert len(x_proj) == len(x)
        total_error += frame_error
    avg_reprojection_error = total_error / len(frames)
    return avg_reprojection_error


def compute_A_B(frames, map_, viewpoints_indices, point_indices):
    N = len(viewpoints_indices)
    A = np.zeros((N, 2, 7))
    B = np.zeros((N, 2, 3))
    for n, (i, j) in enumerate(zip(point_indices, viewpoints_indices)):
        X = map_.X[i][:3].reshape(3, 1)
        A[n] = compute_pose_jacobian_mat(frames[j], X)
        B[n] = compute_X_jacobian_mat(frames[j], X)
    return A, B


def get_indices(frames, map_):
    viewpoints_indices = []
    point_indices = []
    x_t = np.zeros((0, 2))
    x_p = np.zeros((0, 2))
    mask = np.zeros((len(frames), len(map_.X)))
    for frame_n, frame in enumerate(frames):
        frame_x = frame.keypoints[frame.triangulated_idx][:, :2]
        frame_X = map_.X[frame.index_kp_3d]

        mask[frame_n][[frame.index_kp_3d]] = 1
        frame_proj_x = project_3D_to_2D(frame_X, frame.P)[:, :2]

        frame_index = [frame_n] * len(frame.triangulated_idx)
        viewpoints_indices.extend(frame_index)
        point_indices.extend(frame.index_kp_3d)
        assert len(viewpoints_indices) == len(point_indices)
        assert frame_x.shape == frame_proj_x.shape
        x_t = np.vstack((x_t, frame_x))
        x_p = np.vstack((x_p, frame_proj_x))

    assert len(viewpoints_indices) == x_t.shape[0]

    return viewpoints_indices, point_indices, x_t, x_p


def optimize_using_BA(frames, map_):
    visualize_reprojection_error(frames[1], map_)
    avg_error = 1e10
    n_iter = 0
    while avg_error > 500 and n_iter < 10:
        viewpoint_indices, point_indices, x_t, x_p = get_indices(frames, map_)
        optimizer = SBA(viewpoint_indices, point_indices)
        A, B = compute_A_B(frames, map_, viewpoint_indices, point_indices)

        mu = 0.5
        # D = mu * np.identity(b.shape[0])
        # delta = np.linalg.solve(H + D, b)

        del_A, del_B = optimizer.compute(x_t, x_p, A, B, weights=None, mu=mu)
        for f, frame in enumerate(frames):
            if f == 0:
                continue

            Q = rotation_matrix_to_quaternion(frame.R)
            # print(f"old_R_{f}:",frame.R)
            # print(f"old_Q_{f}:",Q)
            # print(f"old_T_{f}:",frame.T)
            # print(f"old_C_{f}:",frame.C)
            frame_updates = del_A[f]
            new_Q = Q - frame_updates[0:4]
            norm = np.linalg.norm(new_Q)
            if np.isclose(norm, 0.0):
                raise ValueError("Cannot normalize quaternion with zero norm")
            new_Q = new_Q / norm
            # print(np.linalg.norm(new_Q))
            new_R = quaternion_to_rotation_matrix(new_Q)
            new_C = frame.C - frame_updates[4:7].reshape(3, 1)
            new_T = -np.dot(new_R, new_C)
            frame.R = new_R
            frame.T = new_T
            frame.compute_projection_matrix()
            # print(f"New_R_{f}:",frame.R)
            # print(f"New_T_{f}:",frame.T)
            # print(f"New_C_{f}:",frame.C)
            # print(f"New_Q_{f}:",new_Q)

        # map_.X[:,:3] = map_.X[:,:3] - del_B
        avg_error = compute_reprojection_error_total(frames, map_)
        print(f"Iteration_{n_iter}: Error: {avg_error}")
        n_iter += 1

    visualize_reprojection_error(frames[1], map_)


# import numpy as np
# from src.utils import get_correspondence_from_file
# from src.frame import Frame
# from src.map import Map
# import cv2
# import os
# import json
# import matplotlib.pyplot as plt
# from src.correspondence import associate_correspondences

# from src.triangulation import triangulate_pts
# from src.map import register_frames_with_map

# from src.utils import project_3D_to_2D
# from src.jacobian import compute_pose_jacobian_mat, compute_X_jacobian_mat
# from src.sparse_ba import SBA, Indices


# def compute_A_B(frames, map_, viewpoints_indices, point_indices):
#     N = len(viewpoints_indices)
#     A = np.zeros((N,2,7))
#     B = np.zeros((N,2,3))
#     print(A.shape)
#     print(B.shape)
#     for n, (i, j) in enumerate(zip(point_indices, viewpoints_indices)):
#         X = map_.X[i][:3].reshape(3,1)
#         A[n] = compute_pose_jacobian_mat(frames[j],X)
#         B[n] = compute_X_jacobian_mat(frames[j], X)
#     return A, B


# def optimize_using_BA(frames, map_):
#     image_folder = "./Stage_1/Stage_14/stage1/box/images"
#     pose_json_path = "./Stage_1/Stage_14/stage1/box/gt_camera_parameters.json"
#     correspondence_folder = "./Stage_1/Stage_14/stage1/box/correspondences"
#     with open(pose_json_path) as f:
#         cam_parameters = json.load(f)
#     K = np.array(cam_parameters["intrinsics"])

#     image_1 = cv2.imread(os.path.join(image_folder, "00000.jpg"))
#     image_2 = cv2.imread(os.path.join(image_folder, "00001.jpg"))
#     image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
#     image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
#     image_idx_1 = 0
#     image_idx_2 = 1
#     frame_1 = Frame(image_1, image_idx_1, K)
#     frame_2 = Frame(image_2, image_idx_2, K)
#     correspondence_file_name = (
#         f"{frame_1.img_id}_{frame_2.img_id}.txt"
#     )
#     correspondence_file_path = os.path.join(
#         correspondence_folder, correspondence_file_name
#     )
#     x_prev, x_curr = get_correspondence_from_file(correspondence_file_path)
#     frame_1.update_keypoints_using_correspondence(x_prev)
#     frame_2.update_keypoints_using_correspondence(x_curr)

#     rt = np.array(cam_parameters["extrinsics"]["00000.jpg"])
#     print(rt)
#     r = rt[:3,:3]
#     t= rt[:3,3]
#     print(r)
#     print(t)
#     frame_1.R = r
#     frame_1.T = t.reshape(3,1)
#     frame_1.compute_projection_matrix()


#     rt = np.array(cam_parameters["extrinsics"]["00001.jpg"])
#     print(rt)
#     r = rt[:3,:3]
#     t= rt[:3,3]
#     print(r)
#     print(t)
#     frame_2.R = r
#     frame_2.T = t.reshape(3,1)
#     frame_2.compute_projection_matrix()

#     associate_correspondences(
#                 frame_1, frame_2
#             )

#     map_ = Map()
#     X = triangulate_pts(
#         x_prev[frame_1.disjoint_idx],
#         x_curr[frame_2.disjoint_idx],
#         frame_1.P,
#         frame_2.P,
#     )
#     colors = [
#         frame_2.image[int(v)][int(u)]
#         for (u, v, _) in x_curr[frame_2.disjoint_idx]
#     ]

#     register_frames_with_map(frame_1, frame_2, map_, X)

#     map_.update_map(X, colors)


#     frames_test = [frame_1, frame_2]

#     assert np.array_equal(frames_test[0].R, frames[0].R)
#     assert np.array_equal(frames_test[0].C, frames[0].C)
#     assert np.array_equal(frames_test[0].T, frames[0].T)
#     assert np.array_equal(frames_test[0].K, frames[0].K)
#     assert np.array_equal(frames_test[0].P, frames[0].P)
#     assert np.array_equal(frames_test[0].RT, frames[0].RT)

#     assert np.array_equal(frames_test[1].R, frames[1].R)
#     assert np.array_equal(frames_test[1].C, frames[1].C)
#     assert np.array_equal(frames_test[1].T, frames[1].T)
#     assert np.array_equal(frames_test[1].K, frames[1].K)
#     assert np.array_equal(frames_test[1].P, frames[1].P)
#     assert np.array_equal(frames_test[1].RT, frames[1].RT)
#     viewpoints_indices = []
#     point_indices = []
#     x_t = np.zeros((0,2))
#     x_p = np.zeros((0,2))
#     mask = np.zeros((len(frames),len(map_.X)))
#     for frame_n,frame in enumerate(frames):
#         frame_x = frame.keypoints[frame.triangulated_idx][:,:2]
#         frame_X = map_.X[frame.index_kp_3d]

#         mask[frame_n][[frame.index_kp_3d]] = 1
#         frame_proj_x = project_3D_to_2D(frame_X, frame.P)[:,:2]

#         frame_index = [frame_n] * len(frame.triangulated_idx)
#         viewpoints_indices.extend(frame_index)
#         point_indices.extend(frame.index_kp_3d)
#         assert len(viewpoints_indices) == len(point_indices)
#         assert frame_x.shape == frame_proj_x.shape
#         x_t = np.vstack((x_t, frame_x))
#         x_p = np.vstack((x_p, frame_proj_x))

#     assert len(viewpoints_indices) == x_t.shape[0]

#     A, B = compute_A_B(frames, map_, viewpoints_indices, point_indices)
#     optimizer = SBA(viewpoints_indices, point_indices)
#     del_A, del_B = optimizer.compute(x_t, x_p, A, B)
#     print("Da:",del_A)
#     print("Db:",del_B)
