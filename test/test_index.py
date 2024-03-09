import numpy as np
from src.correspondence import associate_correspondences
from src.map import Map, register_frames_with_map
from src.frame import Frame
import cv2
import os
import json

image_folder = "./Stage_1/stage1/box/images"
pose_json_path = "./Stage_1/stage1/box/gt_camera_parameters.json"
with open(pose_json_path) as f:
    cam_parameters = json.load(f)
K = np.array(cam_parameters["intrinsics"])

image = cv2.imread(os.path.join(image_folder, "00000.jpg"))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_idx_1 = 0
image_idx_2 = 2
image_idx_3 = 4
frame_1 = Frame(image, image_idx_1, K)
frame_2 = Frame(image, image_idx_2, K)
frame_3 = Frame(image, image_idx_3, K)
frame_1.keypoints = np.array(
    [
        [5, 7, 1],
        [2, 3, 1],
        [4, 3, 1],
        [6, 4, 1],
        [8, 4, 1],
        [9, 10, 1],
        [5, 3, 1],
        [8, 4, 1],
        [10, 7, 1],
    ]
)
frame_2.keypoints = np.array(
    [
        [3, 3, 1],
        [5, 4, 1],
        [1, 3, 1],
        [4, 3, 1],
        [8, 10, 1],
        [11, 2, 1],
        [7, 4, 1],
        [9, 7, 1],
    ]
)
frame_3.keypoints = np.array(
    [[3, 7, 1], [0, 3, 1], [10, 2, 1], [3, 3, 1], [7, 10, 1], [1, 2, 1], [4, 4, 1]]
)

X = np.array(
    [
        [3, 3, 5, 1],
        [5, 4, 8, 1],
        [1, 3, 6, 1],
        [4, 3, 4, 1],
        [8, 10, 3, 1],
        [11, 2, 8, 1],
        [7, 4, 6, 1],
        [9, 7, 2, 1],
    ]
)


def test_update_matches_initial_case():
    frame = Frame(image, image_idx_1, K)
    x_curr = np.array(
        [
            [5, 7, 1],
            [2, 3, 1],
            [4, 3, 1],
            [6, 4, 1],
            [8, 4, 1],
            [9, 10, 1],
            [5, 3, 1],
            [8, 4, 1],
            [10, 7, 1],
        ]
    )
    frame.update_keypoints_using_correspondence(x_curr)
    assert np.array_equal(frame.keypoints, x_curr)
    assert len(frame.matched_idx) == len(x_curr)


def test_update_matches():
    frame = Frame(image, image_idx_1, K)
    frame.keypoints = np.array([[5, 7, 1], [2, 3, 1], [4, 3, 1], [6, 4, 1]])
    frame.matched_idx = [0, 1, 2, 3]
    frame.triangulated_idx = [0, 1, 2, 3]
    x_curr = np.array(
        [[4, 3, 1], [6, 4, 1], [8, 4, 1], [9, 10, 1], [5, 3, 1], [10, 7, 1]]
    )
    frame.update_keypoints_using_correspondence(x_curr)
    assert len(frame.keypoints) == 8
    assert len(frame.matched_idx) == len(x_curr)
    assert np.array_equal(frame.keypoints[frame.matched_idx], x_curr)


def test_splitting_correspondence_initial_case():
    frame_1.matched_idx = [1, 2, 3, 6, 7, 8]
    frame_2.matched_idx = [2, 0, 1, 3, 6, 7]
    frame_1.triangulated_idx = []
    associate_correspondences(frame_1, frame_2)
    assert frame_1.disjoint_idx == [1, 2, 3, 6, 7, 8]
    assert frame_2.disjoint_idx == [2, 0, 1, 3, 6, 7]
    assert frame_1.intersect_idx == []
    assert frame_2.intersect_idx == []


def test_splitting_correspondence_general():
    frame_2.triangulated_idx = [2, 0, 1, 3, 6, 7]
    frame_2.matched_idx = [1, 2, 3, 4, 5]
    frame_3.matched_idx = [6, 1, 3, 4, 2]
    associate_correspondences(frame_2, frame_3)
    assert frame_2.disjoint_idx == [3, 4]
    assert frame_3.disjoint_idx == [3, 4]
    assert frame_2.intersect_idx == [0, 1, 2]
    assert frame_3.intersect_idx == [0, 1, 2]


def test_assigning_3d_index_initial_case():
    frame_2.triangulated_idx = []
    frame_3.index_kp_3d = []
    frame_2.matched_idx = [1, 2, 3, 4, 5]
    frame_3.matched_idx = [6, 1, 3, 4, 2]
    associate_correspondences(frame_2, frame_3)
    assert frame_3.index_kp_3d == []


def test_assigning_3d_index():
    frame_2.triangulated_idx = [2, 0, 1, 3, 6, 7]
    frame_2.matched_idx = [1, 2, 3, 4, 5]
    frame_3.matched_idx = [6, 1, 3, 4, 2]
    associate_correspondences(frame_2, frame_3)
    assert frame_3.index_kp_3d == [2, 0, 3]


def test_linear_pnp_input_generation():
    frame_2.triangulated_idx = [2, 0, 1, 3, 6, 7]
    frame_2.matched_idx = [1, 2, 3, 4, 5]
    frame_3.matched_idx = [6, 1, 3, 4, 2]
    associate_correspondences(frame_2, frame_3)
    idx_3d = frame_3.index_kp_3d
    X_i = X[idx_3d]
    x_i = frame_3.keypoints[np.array(frame_3.matched_idx)[frame_3.intersect_idx]]
    assert len(X_i) == len(x_i)
    assert np.array_equal(X_i, np.array([[1, 3, 6, 1], [3, 3, 5, 1], [4, 3, 4, 1]]))
    assert np.array_equal(x_i, np.array([[4, 4, 1], [0, 3, 1], [3, 3, 1]]))


def test_map_update_initial_case():
    X_new = np.array(
        [
            [3, 3, 5, 1],
            [5, 4, 8, 1],
            [1, 3, 6, 1],
            [4, 3, 4, 1],
            [8, 10, 3, 1],
            [11, 2, 8, 1],
        ]
    )
    color = np.array([[0, 0, 0]])
    colors = np.tile(color, (6, 1))
    map_ = Map()
    frame_prev = Frame(image, image_idx_1, K)
    frame_next = Frame(image, image_idx_2, K)
    frame_prev.triangulated_idx = []
    frame_next.triangulated_idx = []
    frame_prev.disjoint_idx = [0, 1, 2, 3, 4, 5]
    frame_next.disjoint_idx = [0, 1, 2, 3, 4, 5]
    frame_prev.matched_idx = [1, 2, 3, 6, 7, 8]
    frame_next.matched_idx = [2, 0, 1, 3, 6, 7]
    register_frames_with_map(frame_prev, frame_next, map_, X_new)
    map_.update_map(X_new, colors)

    assert (
        len(frame_prev.triangulated_idx)
        == len(frame_next.triangulated_idx)
        == len(X_new)
    )
    assert len(frame_prev.index_kp_3d) == len(frame_next.index_kp_3d) == len(X_new)
    assert frame_prev.triangulated_idx == [1, 2, 3, 6, 7, 8]
    assert frame_next.triangulated_idx == [2, 0, 1, 3, 6, 7]
    assert frame_next.index_kp_3d == [0, 1, 2, 3, 4, 5]
    assert frame_next.index_kp_3d == [0, 1, 2, 3, 4, 5]
    assert np.array_equal(map_.X, X_new)


def test_map_update_general_case():
    X_prev = np.array(
        [
            [3, 3, 5, 1],
            [5, 4, 8, 1],
            [1, 3, 6, 1],
            [4, 3, 4, 1],
            [8, 10, 3, 1],
            [11, 2, 8, 1],
        ]
    )
    color = np.array([[0, 0, 0]])
    colors = np.tile(color, (6, 1))
    map_ = Map()
    map_.update_map(X_prev, colors)
    frame_prev = Frame(image, image_idx_1, K)
    frame_next = Frame(image, image_idx_2, K)
    frame_prev.triangulated_idx = [2, 0, 1, 3, 6, 7]
    frame_next.triangulated_idx = []
    frame_prev.intersect_idx = [0, 1, 2]
    frame_next.intersect_idx = [0, 1, 2]
    frame_prev.disjoint_idx = [3, 4]
    frame_next.disjoint_idx = [3, 4]
    frame_prev.matched_idx = [1, 2, 3, 4, 5]
    frame_next.matched_idx = [6, 1, 3, 4, 2]

    frame_prev.index_kp_3d = [0, 1, 2, 3, 4, 5]
    frame_next.index_kp_3d = [2, 0, 3]

    X_new = np.array([[7, 4, 6, 1], [9, 7, 2, 1]])
    colors = np.tile(color, (2, 1))
    register_frames_with_map(frame_prev, frame_next, map_, X_new)
    map_.update_map(X_new, colors)

    assert len(frame_prev.triangulated_idx) == 8
    assert len(frame_next.triangulated_idx) == 5
    assert frame_prev.triangulated_idx == [2, 0, 1, 3, 6, 7, 4, 5]
    assert frame_next.triangulated_idx == [6, 1, 3, 4, 2]
    assert frame_prev.index_kp_3d == [0, 1, 2, 3, 4, 5, 6, 7]
    assert frame_next.index_kp_3d == [2, 0, 3, 6, 7]
    assert np.array_equal(
        map_.X,
        np.array(
            [
                [3, 3, 5, 1],
                [5, 4, 8, 1],
                [1, 3, 6, 1],
                [4, 3, 4, 1],
                [8, 10, 3, 1],
                [11, 2, 8, 1],
                [7, 4, 6, 1],
                [9, 7, 2, 1],
            ]
        ),
    )
    assert np.array_equal(
        map_.X[frame_next.index_kp_3d],
        np.array(
            [[1, 3, 6, 1], [3, 3, 5, 1], [4, 3, 4, 1], [7, 4, 6, 1], [9, 7, 2, 1]]
        ),
    )
