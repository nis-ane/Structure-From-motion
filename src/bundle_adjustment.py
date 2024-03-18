from src.utils import project_3D_to_2D
from src.visualize import visualize_reprojection_error
import numpy as np
from src.sparse_ba import SBA
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
    """Compute Jacobian matrix splits A and B

    Args:
        frames (list): List of all frames
        map_ (Map): map of 3d points
        viewpoints_indices (list): index containing camera index
        point_indices (list): index containing point index

    Returns:
        A, B: Jacobian Matrices
    """
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
    while avg_error > 100 and n_iter < 50:
        viewpoint_indices, point_indices, x_t, x_p = get_indices(frames, map_)
        optimizer = SBA(viewpoint_indices, point_indices)
        A, B = compute_A_B(frames, map_, viewpoint_indices, point_indices)

        mu = 0.5

        del_A, del_B = optimizer.compute(x_t, x_p, A, B, weights=None, mu=mu)
        for f, frame in enumerate(frames):
            if f == 0:
                continue

            Q = rotation_matrix_to_quaternion(frame.R)
            frame_updates = del_A[f]
            new_Q = Q + frame_updates[0:4]
            norm = np.linalg.norm(new_Q)
            if np.isclose(norm, 0.0):
                raise ValueError("Cannot normalize quaternion with zero norm")
            new_Q = new_Q / norm
            new_R = quaternion_to_rotation_matrix(new_Q)
            new_C = frame.C + 0.1 * frame_updates[4:7].reshape(3, 1)
            new_T = -np.dot(new_R, new_C)
            frame.R = new_R
            frame.T = new_T
            frame.compute_projection_matrix()

        map_.X[:, :3] = map_.X[:, :3] + 0.5 * del_B
        avg_error = compute_reprojection_error_total(frames, map_)
        print(f"Iteration_{n_iter}: Error: {avg_error}")
        n_iter += 1

    visualize_reprojection_error(frames[1], map_)
