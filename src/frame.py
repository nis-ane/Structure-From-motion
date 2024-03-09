import numpy as np
import cv2
from scipy.spatial import distance


class Frame:
    def __init__(self, image, img_id, K, correspondence=False):
        self.image = image
        self.img_id = img_id
        self.keypoints = None
        self.descriptors = []
        self.matched_idx = []
        self.triangulated_idx = []
        self.intersect_idx = []
        self.disjoint_idx = []
        self.index_kp_3d = []
        self.K = K
        self.R = None
        self.T = None
        self.center = None
        self.RT = None
        self.P = None
        self.correspondence = correspondence
        if not self.correspondence:
            self.compute_keypoints_and_descriptors()

    def compute_keypoints_and_descriptors(self, method="sift"):
        """Compute key points and feature descriptors using an specific method

        Args:
            method (str, optional): Detector to detect keypoint. Defaults to 'sift'.
        """

        assert (
            method is not None
        ), "You need to define a feature detection method. Values are: 'sift', 'orb'"
        # detect and extract features from the image
        if method == "sift":
            detector = cv2.SIFT_create()
        elif method == "orb":
            detector = cv2.ORB_create()

        # get keypoints and descriptors
        (keypoints, self.descriptors) = detector.detectAndCompute(self.image, None)
        keypoints_coor = []
        for keypoint in keypoints:
            keypoints_coor.append((keypoint.pt[0], keypoint.pt[1], 1))
        self.keypoints = np.array(keypoints_coor, dtype=np.float32)

    def compute_projection_matrix(self):
        self.RT = np.hstack((self.R, self.T))
        self.P = np.dot(self.K, self.RT)

    def update_keypoints_using_correspondence(self, new_kp):
        if len(self.triangulated_idx) == 0:
            self.keypoints = new_kp
            self.matched_idx = list(np.arange(0, len(new_kp)))
        else:
            dist = distance.cdist(
                self.keypoints[self.triangulated_idx], new_kp, "euclidean"
            )
            min_distances1 = np.min(dist, axis=0)
            idx = np.argmin(dist, axis=0)
            mask = min_distances1 == 0
            new_disjoint = new_kp[~mask]
            self.keypoints = np.vstack((self.keypoints, new_disjoint))

            intersect_idx = idx[mask]
            disjoint_idx = list(
                range(len(self.matched_idx), len(self.matched_idx) + len(new_disjoint))
            )
            self.matched_idx = np.arange(0, len(new_kp))
            self.matched_idx[mask] = intersect_idx
            self.matched_idx[~mask] = disjoint_idx
            self.matched_idx = list(self.matched_idx)

        assert len(new_kp) == len(self.matched_idx)
