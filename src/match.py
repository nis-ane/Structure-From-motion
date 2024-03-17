import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

# Load the images
# img1 = cv2.imread('/Users/hewanshrestha/Desktop/3dcv_project_old/stage2/milk/images/01302.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('/Users/hewanshrestha/Desktop/3dcv_project_old/stage2/milk/images/01359.jpg', cv2.IMREAD_GRAYSCALE)

# # Create a SIFT object
# sift = cv2.SIFT_create()

# # Find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)


def raw_match(descriptors1, descriptors2, dist="euclidean"):
    distances = cdist(descriptors1, descriptors2, dist)
    nearest_neighbors = np.argmin(distances, axis=1)
    matches = np.column_stack((np.arange(descriptors1.shape[0]), nearest_neighbors))
    return matches, distances


def match_with_lowe_first_test(matches, distances, threshold):
    min_distances = distances[np.arange(distances.shape[0]), matches[:, 1]]
    valid_matches = min_distances < threshold
    filtered_matches = matches[valid_matches]
    return filtered_matches


def match_with_lowe_second_test(matches, distances, ratio):
    indices_sorted = np.argsort(distances, axis=1)
    best_distances = distances[
        np.arange(distances.shape[0])[:, np.newaxis], indices_sorted[:, :2]
    ]
    ratio_test_matches = np.nonzero(
        best_distances[:, 0] / best_distances[:, 1] < ratio
    )[0]
    filtered_matches = matches[np.isin(matches[:, 0], ratio_test_matches)]
    return filtered_matches, indices_sorted


def forward_backward_consistency(distances, filtered_index, indices_sorted):
    matched_idx = np.array([], dtype=np.int64)
    for idx in filtered_index:
        forward_match = indices_sorted[idx, 0]
        matched_idx = np.concatenate((matched_idx, np.array([forward_match]))).astype(
            np.int64
        )

    back_matched_idx = np.array([], dtype=np.int64)
    for idx in matched_idx:
        backward_match = np.argmin(distances[:, idx])
        back_matched_idx = np.concatenate(
            (back_matched_idx, np.array([backward_match]))
        ).astype(np.int64)

    filtered_matches = np.where(filtered_index == back_matched_idx)[0]
    first_idx = filtered_index[filtered_matches]
    sec_idx = matched_idx[filtered_matches]
    return first_idx, sec_idx


def match_descriptors(
    descriptors1, descriptors2, dist="euclidean", threshold=0, ratio=0.5
):
    # Compute raw matches
    matches, distances = raw_match(descriptors1, descriptors2, dist)
    print(f"Raw matches: {matches.shape}")

    # Apply Lowe's first test
    matches = match_with_lowe_first_test(matches, distances, threshold)
    print(f"Lowe's first test: {matches.shape}")

    # Apply Lowe's second test
    matches, indices_sorted = match_with_lowe_second_test(matches, distances, ratio)
    print(f"Lowe's second test: {matches.shape}")

    # Apply forward-backward consistency check
    first_idx, sec_idx = forward_backward_consistency(
        distances, matches[:, 0], indices_sorted
    )
    score_idx = np.argsort(distances[first_idx, sec_idx])[::-1]
    matches = np.column_stack((first_idx[score_idx], sec_idx[score_idx]))
    print(f"Forward-backward consistency: {matches.shape}")

    return matches


# # Use the match function
# matches = match_descriptors(
#     descriptors1=des1,
#     descriptors2=des2,
#     dist="euclidean",
#     ratio=0.65,
#     threshold=150,
# )


def ransac_matching(matches, keypoints1, keypoints2, threshold, max_iterations=1000):
    best_inliers = None
    best_model = None

    for iteration in range(max_iterations):
        # Randomly sample a subset of matches (e.g., 2 matches) to estimate the model
        sample = np.random.choice(len(matches), 4, replace=False)
        src_pts = np.float32([keypoints1[matches[idx, 0]][:2] for idx in sample])
        dst_pts = np.float32([keypoints2[matches[idx, 1]][:2] for idx in sample])

        # Check if the points are too close together or aligned vertically/horizontally
        if (
            np.linalg.norm(src_pts[0] - src_pts[1]) < 1e-6
            or np.abs(src_pts[0, 0] - src_pts[1, 0]) < 1e-6
        ):
            continue

        # Estimate the line parameters (slope and intercept) using the sampled matches
        slope, intercept = np.polyfit(src_pts[:, 0], dst_pts[:, 0], 1)

        # Apply the estimated model to all matches
        src_pts_all = np.float32([keypoints1[idx][:2] for idx in matches[:, 0]])
        predicted_pts = slope * src_pts_all[:, 0] + intercept

        # Compute distances between predicted points and actual points in the second image
        dst_pts_all = np.float32([keypoints2[idx][:2] for idx in matches[:, 1]])
        distances = np.abs(predicted_pts - dst_pts_all[:, 0])

        # Count inliers based on the threshold
        inliers = matches[distances < threshold]

        # Update the best model if the current model has more inliers
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (slope, intercept)

    return best_inliers


# inliers = ransac_matching(matches, kp1, kp2, threshold=50, max_iterations=1000)
# print(inliers.shape)

# Load the inliers from the correspondence file
# inliers = np.loadtxt('inliers.txt', dtype=int)

# Draw matches manually using Matplotlib
# fig, ax = plt.subplots(figsize=(20.0, 20.0))
# ax.imshow(np.hstack((img1, img2)), cmap='gray')

# k = 5

# for i, match in enumerate(inliers[:k, :]):
#     idx1, idx2 = match
#     pt1 = (int(kp1[idx1].pt[0]), int(kp1[idx1].pt[1]))
#     pt2 = (int(kp2[idx2].pt[0] + img1.shape[1]), int(kp2[idx2].pt[1]))

#     # Choose a different color for each match
#     color = plt.cm.nipy_spectral(i / k)
#     ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
#     ax.plot(pt1[0], pt1[1], 'yo')  # plot keypoint in the first image
#     ax.plot(pt2[0], pt2[1], 'ro')  # plot keypoint in the second image

# plt.show()
