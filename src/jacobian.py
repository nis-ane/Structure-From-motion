"""
This script should contain the code for
Computing Jacobian matrix
1. Derivative w.r..t X
2. Derivative w.r..t C
3. Derivative w.r..t R
4. Derivative w.r..t Q(Quaternion)
5. Compute Jacobian Matrix
6. Get A and B matrix -> Splitting Jacobian matrix into camera parameters and 3d points
"""
import numpy as np
from src.utils import (
    project_3D_to_2D,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve


def compute_C_jacobian_mat(frame, X):
    fx = frame.K[0, 0]
    fy = frame.K[1, 1]
    ox = frame.K[0, 2]
    oy = frame.K[1, 2]
    R = frame.R
    C = frame.C

    u = np.dot(
        np.array(
            [
                fx * R[0, 0] + ox * R[2, 0],
                fx * R[0, 1] + ox * R[2, 1],
                fx * R[0, 2] + ox * R[2, 2],
            ]
        ),
        (X - C),
    )
    v = np.dot(
        np.array(
            [
                fy * R[1, 0] + oy * R[2, 0],
                fy * R[1, 1] + oy * R[2, 1],
                fy * R[1, 2] + oy * R[2, 2],
            ]
        ),
        (X - C),
    )
    w = np.dot(np.array([R[2, 0], R[2, 1], R[2, 2]]), (X - C))

    du_dC = -np.array(
        [
            fx * R[0, 0] + ox * R[2, 0],
            fx * R[0, 1] + ox * R[2, 1],
            fx * R[0, 2] + ox * R[2, 2],
        ]
    )
    dv_dC = -np.array(
        [
            fy * R[1, 0] + oy * R[2, 0],
            fy * R[1, 1] + oy * R[2, 1],
            fy * R[1, 2] + oy * R[2, 2],
        ]
    )
    dw_dC = -np.array([R[2, 0], R[2, 1], R[2, 2]])

    dF_dC = np.array(
        [(w * du_dC - u * dw_dC) / w ** 2, (w * dv_dC - v * dw_dC) / w ** 2]
    )

    assert dF_dC.shape == (2, 3)
    return dF_dC


def compute_X_jacobian_mat(frame, X):
    fx = frame.K[0, 0]
    fy = frame.K[1, 1]
    ox = frame.K[0, 2]
    oy = frame.K[1, 2]
    R = frame.R
    C = frame.C

    u = np.dot(
        np.array(
            [
                fx * R[0, 0] + ox * R[2, 0],
                fx * R[0, 1] + ox * R[2, 1],
                fx * R[0, 2] + ox * R[2, 2],
            ]
        ),
        (X - C),
    )
    v = np.dot(
        np.array(
            [
                fy * R[1, 0] + oy * R[2, 0],
                fy * R[1, 1] + oy * R[2, 1],
                fy * R[1, 2] + oy * R[2, 2],
            ]
        ),
        (X - C),
    )
    w = np.dot(np.array([R[2, 0], R[2, 1], R[2, 2]]), (X - C))

    du_dX = np.array(
        [
            fx * R[0, 0] + ox * R[2, 0],
            fx * R[0, 1] + ox * R[2, 1],
            fx * R[0, 2] + ox * R[2, 2],
        ]
    )
    dv_dX = np.array(
        [
            fy * R[1, 0] + oy * R[2, 0],
            fy * R[1, 1] + oy * R[2, 1],
            fy * R[1, 2] + oy * R[2, 2],
        ]
    )
    dw_dX = np.array([R[2, 0], R[2, 1], R[2, 2]])

    dF_dX = np.array(
        [(w * du_dX - u * dw_dX) / w ** 2, (w * dv_dX - v * dw_dX) / w ** 2]
    )

    assert dF_dX.shape == (2, 3)
    return dF_dX


def compute_R_jacobian_mat(frame, X):
    fx = frame.K[0, 0]
    fy = frame.K[1, 1]
    ox = frame.K[0, 2]
    oy = frame.K[1, 2]
    R = frame.R
    C = frame.C

    u = np.dot(
        np.array(
            [
                fx * R[0, 0] + ox * R[2, 0],
                fx * R[0, 1] + ox * R[2, 1],
                fx * R[0, 2] + ox * R[2, 2],
            ]
        ),
        (X - C),
    )
    v = np.dot(
        np.array(
            [
                fy * R[1, 0] + oy * R[2, 0],
                fy * R[1, 1] + oy * R[2, 1],
                fy * R[1, 2] + oy * R[2, 2],
            ]
        ),
        (X - C),
    )
    w = np.dot(np.array([R[2, 0], R[2, 1], R[2, 2]]), (X - C))

    du_dR = np.hstack((np.hstack((fx * (X - C).T, np.zeros((1, 3)))), ox * (X - C).T))
    dv_dR = np.hstack((np.hstack((np.zeros((1, 3)), fy * (X - C).T)), oy * (X - C).T))
    dw_dR = np.hstack((np.hstack((np.zeros((1, 3)), np.zeros((1, 3)))), (X - C).T))

    dF_dR = np.vstack(
        ((w * du_dR - u * dw_dR) / w ** 2, (w * dv_dR - v * dw_dR) / w ** 2)
    )

    assert dF_dR.shape == (2, 9)
    return dF_dR


def compute_dR_dQ_mat(Q):
    # Q = [w,x,y,z]  but the derivative is returned w.r.t [x,y,z,w]
    Q_x = Q[0]
    Q_y = Q[1]
    Q_z = Q[2]
    Q_w = Q[3]

    dR11_dQ = np.array([0, -4 * Q_y, -4 * Q_z, 0])
    dR12_dQ = np.array([2 * Q_y, 2 * Q_x, -2 * Q_w, 2 * Q_z])
    dR13_dQ = np.array([2 * Q_z, 2 * Q_w, 2 * Q_x, 2 * Q_y])
    dR21_dQ = np.array([2 * Q_y, 2 * Q_x, 2 * Q_w, 2 * Q_z])
    dR22_dQ = np.array([-4 * Q_x, 0, -4 * Q_z, 0])
    dR23_dQ = np.array([-2 * Q_w, 2 * Q_z, 2 * Q_y, 2 * Q_x])
    dR31_dQ = np.array([2 * Q_z, -2 * Q_w, -2 * Q_x, -2 * Q_y])
    dR32_dQ = np.array([2 * Q_w, 2 * Q_z, 2 * Q_y, 2 * Q_x])
    dR33_dQ = np.array([-4 * Q_x, -4 * Q_y, 0, 0])

    dR_dQ = np.vstack(
        [
            dR11_dQ,
            dR12_dQ,
            dR13_dQ,
            dR21_dQ,
            dR22_dQ,
            dR23_dQ,
            dR31_dQ,
            dR32_dQ,
            dR33_dQ,
        ]
    )

    assert dR_dQ.shape == (9, 4)
    return dR_dQ


def compute_Q_jacobian_mat(frame, X):
    Q = rotation_matrix_to_quaternion(frame.R)
    dF_dR = compute_R_jacobian_mat(frame, X)
    dR_dQ = compute_dR_dQ_mat(Q)
    dF_dQ = np.dot(dF_dR, dR_dQ)
    assert dF_dQ.shape == (2, 4)
    return dF_dQ


def compute_pose_jacobian_mat(frame, X):
    dF_dQ = compute_Q_jacobian_mat(frame, X)
    dF_dC = compute_C_jacobian_mat(frame, X)
    J_p = np.hstack((dF_dQ, dF_dC))
    assert J_p.shape == (2, 7)
    return J_p


def compute_jacobian(frames, map_):
    N = len(map_.X)
    F = len(frames)
    J = np.zeros((N * F * 2, N * 3 + F * 7))
    for n in range(N):
        X = map_.X[0][:3].reshape(3, 1)
        for f in range(F):
            frame = frames[f]
            if n in frame.index_kp_3d:
                J_p = compute_pose_jacobian_mat(frame, X)
                J_x = compute_X_jacobian_mat(frame, X)

                J[
                    n * 2 * F + 2 * f : n * 2 * F + 2 * (f + 1), 7 * f : 7 * (f + 1)
                ] = J_p
                J[
                    n * 2 * F + 2 * f : n * 2 * F + 2 * (f + 1),
                    7 * F + 3 * n : 7 * F + 3 * (n + 1),
                ] = J_x
            else:
                continue
    return J


# def compute_A_B(frames, map_):
#     N = len(map_.X)
#     F = len(frames)

#     J = np.zeros((N * F * 2, N * 3 + F * 7))
#     A = np.zeros((N * 2, F * 7))
#     B = np.zeros((N * 2, F * 3))

#     for n in range(N):
#         X = map_.X[n][:3].reshape(3, 1)
#         for f in range(F):
#             frame = frames[f]
#             if n in frame.index_kp_3d:
#                 A[2 * n : 2 * (n + 1), 7 * f : 7 * (f + 1)] = compute_pose_jacobian_mat(
#                     frame, X
#                 )
#                 B[2 * n : 2 * (n + 1), 3 * f : 3 * (f + 1)] = compute_X_jacobian_mat(
#                     frame, X
#                 )

#             else:
#                 continue

#     return A, B


def compute_A_B(frames, map_, viewpoints_indices, point_indices):
    N = len(viewpoints_indices)
    A = np.zeros((N, 2, 7))
    B = np.zeros((N, 2, 3))
    print(A.shape)
    print(B.shape)
    for n, (i, j) in enumerate(zip(point_indices, viewpoints_indices)):
        X = map_.X[i][:3].reshape(3, 1)
        A[n] = compute_pose_jacobian_mat(frames[j], X)
        B[n] = compute_X_jacobian_mat(frames[j], X)
    return A, B


def compute_reprojection_errors(frames, map_):
    N = len(map_.X)
    F = len(frames)
    epsilon = np.zeros((F, N, 2))
    for f in range(F):
        X = map_.X[frames[f].index_kp_3d]
        x_proj = project_3D_to_2D(X, frames[f].P)
        x = frames[f].keypoints[frames[f].triangulated_idx]
        epsilon[f, frames[f].index_kp_3d, 0] = (x - x_proj)[:, 0]
        epsilon[f, frames[f].index_kp_3d, 1] = (x - x_proj)[:, 1]
        print("Epsilon:", epsilon.shape)
    return epsilon


def calc_epsilon_a(A, epsilon, N, F):
    epsilon_a = np.zeros((7, F))
    for f in range(F):
        epsilon_a_j = np.zeros((7, 1))
        for n in range(N):
            A_nf = A[2 * n : 2 * (n + 1), 7 * f : 7 * (f + 1)]
            AT_e = np.dot(A_nf.T, epsilon[f, n].reshape(2, 1))
            epsilon_a_j = epsilon_a_j + AT_e
        epsilon_a[:, f] = epsilon_a_j.reshape(-1)
    return epsilon_a


def calc_epsilon_b(B, epsilon, N, F):
    epsilon_b = np.zeros((3, N))
    for n in range(N):
        epsilon_b_i = np.zeros((3, 1))
        for f in range(F):
            B_nf = B[2 * n : 2 * (n + 1), 3 * f : 3 * (f + 1)]
            BT_e = np.dot(B_nf.T, epsilon[f, n].reshape(2, 1))
            epsilon_b_i = epsilon_b_i + BT_e
        epsilon_b[:, n] = epsilon_b_i.reshape(-1)
    return epsilon_b


def compute_U_V_W_Y_ea_eb(frames, map_):
    N = len(map_.X)
    F = len(frames)

    epsilon = compute_reprojection_errors(frames, map_)
    A, B = compute_A_B(frames, map_)

    U = np.zeros((7, F * 7))
    V = np.zeros((3, N * 3))
    W = np.zeros((F * 7, N * 3))
    Y = np.zeros((F * 7, N * 3))

    for f in range(F):
        U_f = np.zeros((7, 7))
        for n in range(N):
            A_nf = A[2 * n : 2 * (n + 1), 7 * f : 7 * (f + 1)]
            AT_A = np.dot(A_nf.T, A_nf)
            U_f = U_f + AT_A
        U[:, 7 * f : 7 * (f + 1)] = U_f

    for n in range(N):
        V_n = np.zeros((3, 3))
        for f in range(F):
            B_nf = B[2 * n : 2 * (n + 1), 3 * f : 3 * (f + 1)]
            BT_B = np.dot(B_nf.T, B_nf)
            V_n = V_n + BT_B
        V[:, 3 * n : 3 * (n + 1)] = V_n

    for n in range(N):
        for f in range(F):
            A_nf = A[2 * n : 2 * (n + 1), 7 * f : 7 * (f + 1)]
            B_nf = B[2 * n : 2 * (n + 1), 3 * f : 3 * (f + 1)]
            AT_B = np.dot(A_nf.T, B_nf)
            W_nf = AT_B
            W[7 * f : 7 * (f + 1), 3 * n : 3 * (n + 1)] = W_nf

    for n in range(N):
        V_n = V[:, 3 * n : 3 * (n + 1)]
        for f in range(F):
            W_nf = W[7 * f : 7 * (f + 1), 3 * n : 3 * (n + 1)]
            Y_nf = np.dot(W_nf, np.linalg.inv(V_n))
            Y[7 * f : 7 * (f + 1), 3 * n : 3 * (n + 1)] = Y_nf

    e_a = calc_epsilon_a(A, epsilon, N, F)
    e_b = calc_epsilon_b(B, epsilon, N, F)

    return U, V, W, Y, e_a, e_b


def compute_epsilon_j(ea, eb, Y, F, N):
    ej = np.zeros((7 * F))
    for f in range(F):
        Yeb = np.zeros((7, 1))
        for n in range(N):
            Ynf = Y[7 * f : 7 * (f + 1), 3 * n : 3 * (n + 1)]
            Yeb = Yeb + np.dot(Ynf, eb[:, n].reshape(3, 1))
        ej[7 * f : 7 * (f + 1)] = ea[:, f] - Yeb.reshape(-1)
    return ej


def compute_S_mat(U, W, Y, F, N):
    S = np.zeros((F * 7, F * 7))
    print(S.shape)
    for j in range(F):
        U_j = U[:, 7 * j : 7 * (j + 1)]
        for k in range(F):
            YjWk = np.zeros((7, 7))
            for n in range(N):
                Ynj = Y[7 * j : 7 * (j + 1), 3 * n : 3 * (n + 1)]
                Wnk = W[7 * k : 7 * (k + 1), 3 * n : 3 * (n + 1)]
                YjWk = YjWk + np.dot(Ynj, Wnk.T)
            if j == k:
                S_jk = -YjWk + U_j
            else:
                S_jk = -YjWk
            S[7 * j : 7 * (j + 1), 7 * k : 7 * (k + 1)] = S_jk
    return S


def compute_delta_pose(S, e_j):
    del_pose = spsolve(S, e_j)
    return del_pose


def compute_delta_X(del_a, V, W, eb, F, N):
    del_X = np.zeros(3 * N)
    for n in range(N):
        Wf_del_af = np.zeros((3, 1))
        for f in range(F):
            Wnf = W[7 * f : 7 * (f + 1), 3 * n : 3 * (n + 1)]
            Wf_del_af = Wf_del_af + np.dot(Wnf.T, del_a[7 * f : 7 * (f + 1)]).reshape(
                3, 1
            )
        del_X[3 * n : 3 * (n + 1)] = np.dot(
            np.linalg.inv(V[:, 3 * n : 3 * (n + 1)]), (eb[:, n] - Wf_del_af.reshape(-1))
        )
    return del_X


def optimize_using_BA(frames, map_):
    F = len(frames)
    N = len(map_.X)
    U, V, W, Y, e_a, e_b = compute_U_V_W_Y_ea_eb(frames, map_)
    e_j = compute_epsilon_j(e_a, e_b, Y, F, N)
    S = compute_S_mat(U, W, Y, F, N)
    del_pose = compute_delta_pose(S, e_j)
    del_X = compute_delta_X(del_pose, V, W, e_b, F, N)

    for f, frame in enumerate(frames):
        Q = rotation_matrix_to_quaternion(frame.R)
        frame_updates = del_pose[7 * f : 7 * (f + 1)]
        new_Q = Q + frame_updates[0:4]
        norm = np.linalg.norm(new_Q)
        if np.isclose(norm, 0.0):
            raise ValueError("Cannot normalize quaternion with zero norm")
        new_Q = new_Q / norm

        new_R = quaternion_to_rotation_matrix(new_Q)
        new_C = frame.C + frame_updates[4:7].reshape(3, 1)
        new_T = -np.dot(new_R, new_C)
        frame.R = new_R
        frame.T = new_T
        frame.compute_projection_matrix()

    map_.X[:, :3] = map_.X[:, :3] - del_X.reshape(len(map_.X), 3)
