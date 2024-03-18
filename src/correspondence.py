# """This script is used for retriving correspondece for a pair of images
# 1. Given 3d points and 2d correspondence generate 3d-2d correspondence for new frame
# """
# import numpy as np
# from scipy.spatial import distance


# def get_2d_to_2d_correspondence(frame_1, frame_2):
#     # Compute matches from
#     # Given:, frame_1.keypoints, frame_1.descriptor
#     # Given: frame_2.keypoints, frame_1.descriptor
#     # set matched idx for frame_1 and frame_2.
#     # frame_1.keypoints = [a,b,c,d,e] frame_2.keypoints = [a,c,b,f,g,h]
#     # Ratio test
#     # Forward backward consistency
#     # Ransac -> _, inlier = Estimate essential mat(x1, x2)

#     # x_prev = [a,b,c] x_curr = [a,b,c]
#     # frame_1.matched_idx = [0,1,2]  Set this
#     # frame_2.matched_idx = [0,2,1]  set this
#     # return x_prev, x_curr
#     pass


# def get_3d_to_2d_correspondence(X, x1_prev, x1_curr, x2_curr):
#     assert len(X) == len(
#         x1_prev
#     ), "for previous correspondence Respective 3D points are not available"
#     assert len(x1_curr) == len(x2_curr)
#     dist = distance.cdist(x1_prev, x1_curr, "euclidean")
#     min_distances1 = np.min(dist, axis=0)
#     idx = np.argmin(dist, axis=0)

#     mask = min_distances1 == 0
#     idx_f = idx[mask]
#     x1_curr_f = x1_curr[mask]
#     x2_curr_f = x2_curr[mask]
#     X_f = X[idx_f]

#     assert len(x2_curr_f) == len(X_f)
#     return X_f, x2_curr_f


# def associate_correspondences(frame_prev, frame_curr):
#     if len(frame_prev.triangulated_idx) == 0:
#         frame_prev.disjoint_idx = frame_prev.matched_idx
#         frame_curr.disjoint_idx = frame_curr.matched_idx

#     else:
#         traingulated_pts = frame_prev.keypoints[frame_prev.triangulated_idx]
#         new_match = frame_prev.keypoints[frame_prev.matched_idx]
#         dist = distance.cdist(traingulated_pts, new_match, "euclidean")
#         min_distances1 = np.min(dist, axis=0)
#         idx = np.argmin(dist, axis=0)
#         mask = min_distances1 == 0
#         masked_idx = idx[mask]

#         frame_prev.intersect_idx = list(np.arange(0, len(new_match))[mask])
#         frame_prev.disjoint_idx = list(np.arange(0, len(new_match))[~mask])
#         frame_curr.intersect_idx = list(np.arange(0, len(new_match))[mask])
#         frame_curr.disjoint_idx = list(np.arange(0, len(new_match))[~mask])

#         frame_curr.index_kp_3d = list(masked_idx)


"""This script is used for retriving correspondece for a pair of images
1. Given 3d points and 2d correspondence generate 3d-2d correspondence for new frame
"""
import numpy as np
from scipy.spatial import distance
from src.match import match_descriptors, ransac_matching
import cv2

RATIO_TEST_1_THRESH = 100
RATIO_TEST_2_THRESH = 0.5
RANSAC_THRESH = 50


def get_2d_to_2d_correspondence(frame_1, frame_2):
    # Compute matches from
    # Given:, frame_1.keypoints, frame_1.descriptor
    # Given: frame_2.keypoints, frame_1.descriptor
    # set matched idx for frame_1 and frame_2.
    # frame_1.keypoints = [a,b,c,d,e] frame_2.keypoints = [a,c,b,f,g,h]
    # Ratio test
    # Forward backward consistency
    # Ransac -> _, inlier = Estimate essential mat(x1, x2)

    # x_prev = [a,b,c] x_curr = [a,b,c]
    # frame_1.matched_idx = [0,1,2]  Set this
    # frame_2.matched_idx = [0,2,1]  set this
    # return x_prev, x_curr

    matches = match_descriptors(
        frame_1.descriptors,
        frame_2.descriptors,
        dist="euclidean",
        threshold=RATIO_TEST_1_THRESH,
        ratio=RATIO_TEST_2_THRESH,
    )

    print(matches.shape)
    print(matches)

    if isinstance(frame_1.keypoints[0], cv2.KeyPoint):
        print("frame_1.keypoints[0] is a cv2.KeyPoint object")
    else:
        print("frame_1.keypoints[0] is not a cv2.KeyPoint object")

    inliers = ransac_matching(
        matches,
        frame_1.keypoints,
        frame_2.keypoints,
        threshold=RANSAC_THRESH,
        max_iterations=1000,
    )
    print(inliers.shape)

    frame_1.matched_idx = list(inliers[:, 0])
    frame_2.matched_idx = list(inliers[:, 1])

    x_prev = frame_1.keypoints[frame_1.matched_idx]
    print(x_prev.shape)
    x_curr = frame_2.keypoints[frame_2.matched_idx]
    print(x_curr.shape)

    return x_prev, x_curr


def get_3d_to_2d_correspondence(X, x1_prev, x1_curr, x2_curr):
    assert len(X) == len(
        x1_prev
    ), "for previous correspondence Respective 3D points are not available"
    assert len(x1_curr) == len(x2_curr)
    dist = distance.cdist(x1_prev, x1_curr, "euclidean")
    min_distances1 = np.min(dist, axis=0)
    idx = np.argmin(dist, axis=0)

    mask = min_distances1 == 0
    idx_f = idx[mask]
    x1_curr_f = x1_curr[mask]
    x2_curr_f = x2_curr[mask]
    X_f = X[idx_f]

    assert len(x2_curr_f) == len(X_f)
    return X_f, x2_curr_f


def associate_correspondences(frame_prev, frame_curr):
    if len(frame_prev.triangulated_idx) == 0:
        frame_prev.disjoint_idx = list(np.arange(0, len(frame_prev.matched_idx)))
        frame_curr.disjoint_idx = list(np.arange(0, len(frame_curr.matched_idx)))

    else:
        traingulated_pts = frame_prev.keypoints[frame_prev.triangulated_idx]
        new_match = frame_prev.keypoints[frame_prev.matched_idx]
        dist = distance.cdist(traingulated_pts, new_match, "euclidean")
        min_distances1 = np.min(dist, axis=0)
        idx = np.argmin(dist, axis=0)
        mask = min_distances1 == 0
        masked_idx = idx[mask]

        assert len(new_match) == len(frame_prev.matched_idx)
        frame_prev.intersect_idx = list(np.arange(0, len(new_match))[mask])
        frame_prev.disjoint_idx = list(np.arange(0, len(new_match))[~mask])
        frame_curr.intersect_idx = list(np.arange(0, len(new_match))[mask])
        frame_curr.disjoint_idx = list(np.arange(0, len(new_match))[~mask])

        if len(frame_prev.disjoint_idx) > 0:
            assert max(frame_prev.disjoint_idx) < len(frame_prev.matched_idx)
        if len(frame_prev.intersect_idx) > 0:
            assert max(frame_prev.intersect_idx) < len(frame_prev.matched_idx)
        if len(frame_curr.disjoint_idx) > 0:
            assert max(frame_curr.disjoint_idx) < len(frame_curr.matched_idx)
        if len(frame_curr.intersect_idx) > 0:
            assert max(frame_curr.intersect_idx) < len(frame_curr.matched_idx)

        frame_curr.index_kp_3d = list(masked_idx)
