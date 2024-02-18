import numpy as np
from src.pose_estimation import (
    estimate_pose_Linear_PnP,
    estimate_pose_Linear_PnP_RANSAC,
    cleanup_RT_mat,
)
from src.utils import project_3D_to_2D


def test_cleanup_RT():
    R = np.array(
        [
            [8.00594113e-04, 2.07275507e-15, 2.21756931e-13],
            [9.63817239e-15, 1.71564047e-10, -8.00594199e-04],
            [-2.49159865e-14, 8.00594113e-04, 3.76291759e-13],
        ]
    )
    T = np.array([0.00020015, 0.00020015, -0.00020015])
    R_c, T_c = cleanup_RT_mat(R, T)
    _, d, _ = np.linalg.svd(R)
    assert np.array_equal(T_c, T / d[0])
    assert np.round(np.linalg.det(R_c)) == 1


def test_linear_pnp_using_crafted_data():
    X = np.array(
        [
            [0.0, 5.0, 2.0, 1.0],
            [2.0, 5.0, 3.0, 1.0],
            [2.0, 4.0, 2.0, 1.0],
            [3.0, 5.0, 2.0, 1.0],
            [2.0, 4.0, 3.0, 1.0],
            [3.0, 4.0, 2.0, 1.0],
        ]
    )

    K = np.array(
        [
            [700.0000, 0.0000, 320.0000],
            [0.0000, 933.3333, 240.0000],
            [0.0000, 0.0000, 1.0000],
        ]
    )

    RT_gt = np.array(
        [
            [1.0000, 0.0000, 0.0000, 0.2500],
            [0.0000, 0.0000, -1.0000, 0.2500],
            [0.0000, 1.0000, 0.0000, -0.2500],
        ]
    )

    P_gt = np.dot(K, RT_gt)
    x = project_3D_to_2D(X, P_gt)

    R, T = estimate_pose_Linear_PnP(x, X, K)

    RT = np.hstack((R, T))
    P = np.dot(K, RT)
    x_pred = project_3D_to_2D(X, P)

    assert np.linalg.norm(x_pred - x) < 0.05
    assert np.array_equal(np.round(RT_gt, 2), np.round(RT, 2))


def test_linear_pnp_ransac_using_crafted_data():

    X = np.array(
        [
            [0.0, 5.0, 2.0, 1.0],
            [2.0, 5.0, 3.0, 1.0],
            [2.0, 4.0, 2.0, 1.0],
            [3.0, 5.0, 2.0, 1.0],
            [2.0, 4.0, 3.0, 1.0],
            [3.0, 4.0, 2.0, 1.0],
        ]
    )

    K = np.array(
        [
            [700.0000, 0.0000, 320.0000],
            [0.0000, 933.3333, 240.0000],
            [0.0000, 0.0000, 1.0000],
        ]
    )

    RT_gt = np.array(
        [
            [1.0000, 0.0000, 0.0000, 0.2500],
            [0.0000, 0.0000, -1.0000, 0.2500],
            [0.0000, 1.0000, 0.0000, -0.2500],
        ]
    )
    P_gt = np.dot(K, RT_gt)
    x = project_3D_to_2D(X, P_gt)

    R, T, _ = estimate_pose_Linear_PnP_RANSAC(x, X, K)

    RT = np.hstack((R, T))
    P = np.dot(K, RT)
    x_pred = project_3D_to_2D(X, P)

    assert np.linalg.norm(x_pred - x) < 0.05
    assert np.array_equal(np.round(RT_gt, 2), np.round(RT, 2))
