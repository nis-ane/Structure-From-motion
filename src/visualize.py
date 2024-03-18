import trimesh
import numpy as np
import matplotlib.pyplot as plt
from src.utils import *
import cv2
import plyfile

view_angle = [0, 0, 0]
distance = 100
cam_center = [0.5, 0.5, 0.5]


def get_poses_obj(poses, size=0.1, color=[0.0, 0.0, 0.0, 1.0]):
    objects = []

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array(
            [
                [pos, a],
                [pos, b],
                [pos, c],
                [pos, d],
                [a, b],
                [b, c],
                [c, d],
                [d, a],
                [pos, o],
            ]
        )

        segs_temp = trimesh.load_path(segs)  # , colors = colors)
        colors = np.tile(np.array(color), (len(segs_temp.entities), 1))
        segs = trimesh.load_path(segs, colors=colors)
        print()
        objects.append(segs)
    return objects


def visualise_pose_and_3d_points(poses, pointcloud, colors=None):
    poses_obj = get_poses_obj(poses)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=pointcloud, colors=colors))
    scene.add_geometry(poses_obj)
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.set_camera(angles=view_angle, distance=distance, center=cam_center)
    scene.show()


def visualise_3d_points(points_3d, colors=None):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points_3d, colors=colors))
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.set_camera(angles=view_angle, distance=distance, center=cam_center)
    scene.show()


def visualise_poses(poses, color=[0.0, 0.0, 0.0, 1.0]):
    poses_obj = get_poses_obj(poses, color=color)
    scene = trimesh.Scene()
    scene.add_geometry(poses_obj)
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.set_camera(angles=view_angle, distance=distance, center=cam_center)
    scene.show()


def visualise_poses_and_3d_points_with_gt(
    poses, pointcloud, cam_parameters, n=None, colors=None
):
    poses_dict = cam_parameters["extrinsics"]
    sorted_poses_dict = dict(sorted(poses_dict.items()))
    count = 1
    poses_gt = []
    for _, value in sorted_poses_dict.items():
        RT = np.array(value)[:3, :]
        poses_gt.append(RT)
        if n is None:
            continue
        count = count + 1
        if count > n:
            break

    poses_obj = get_poses_obj(poses, color=[0.0, 0.0, 0.0, 1.0])
    poses_gt_obj = get_poses_obj(poses_gt, color=[0.0, 1.0, 0.0, 1.0])
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.add_geometry(poses_gt_obj)
    scene.add_geometry(poses_obj)
    scene.add_geometry(trimesh.PointCloud(vertices=pointcloud, colors=colors))
    scene.set_camera(angles=view_angle, distance=distance, center=cam_center)
    scene.show()


def visualise_poses_with_gt(poses, cam_parameters, n=None):
    poses_dict = cam_parameters["extrinsics"]
    sorted_poses_dict = dict(sorted(poses_dict.items()))
    count = 1
    poses_gt = []
    for _, value in sorted_poses_dict.items():
        RT = np.array(value)[:3, :]
        poses_gt.append(RT)
        if n is None:
            continue
        count = count + 1
        if count > n:
            break

    poses_obj = get_poses_obj(poses, color=[0.0, 0.0, 0.0, 1.0])
    poses_gt_obj = get_poses_obj(poses_gt, color=[0.0, 1.0, 0.0, 1.0])
    scene = trimesh.Scene()
    scene.add_geometry(poses_gt_obj)
    scene.add_geometry(poses_obj)
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.set_camera(angles=view_angle, distance=distance, center=cam_center)
    scene.show()


def visualise_gt_poses(cam_parameters, n=None):
    poses_dict = cam_parameters["extrinsics"]
    sorted_poses_dict = dict(sorted(poses_dict.items()))
    count = 1
    poses = []
    for _, value in sorted_poses_dict.items():
        RT = np.array(value)[:3, :]
        poses.append(RT)
        if n is None:
            continue
        count = count + 1
        if count > n:
            break

    visualise_poses(poses, color=[0.0, 1.0, 0.0, 1.0])


def visualize_correspondences(frame_1, frame_2):
    matching_result = cv2.drawMatches(
        frame_1.image,
        frame_1.keypoints,
        frame_2.image,
        frame_2.keypoints,
        matches,
        None,
        flags=2,
    )
    pass


def visualize_pointcloud(path):
    plydata = plyfile.PlyData.read(path)

    # Extract x, y, z coordinates
    x = plydata["vertex"]["x"]
    y = plydata["vertex"]["y"]
    z = plydata["vertex"]["z"]

    r = plydata["vertex"]["red"]
    g = plydata["vertex"]["green"]
    b = plydata["vertex"]["blue"]

    new_x = x[abs(x) < 100]
    new_y = y[abs(x) < 100]
    new_z = z[abs(x) < 100]

    r = r[abs(x) < 100]
    g = g[abs(x) < 100]
    b = b[abs(x) < 100]

    print(np.mean(new_x))
    print(np.mean(new_y))
    print(np.mean(new_z))

    print(np.max(new_x))
    print(np.max(new_y))
    print(np.max(new_z))

    print(np.min(new_x))
    print(np.min(new_y))
    print(np.min(new_z))

    print(np.percentile(new_x, 15))
    print(np.percentile(new_y, 15))
    print(np.percentile(new_z, 15))

    print(np.percentile(new_x, 85))
    print(np.percentile(new_y, 85))
    print(np.percentile(new_z, 85))

    pointcloud = np.stack([new_x, new_y, new_z], axis=-1)
    colors = np.stack([r, g, b], axis=-1)
    visualise_3d_points(pointcloud, colors)


def visualize_reprojection_error(frame, map_):
    X = map_.X[frame.index_kp_3d]
    x_proj = project_3D_to_2D(X, frame.P)
    x_proj[:, 0] = np.clip(x_proj[:, 0], 0, frame.image.shape[1])
    x_proj[:, 1] = np.clip(x_proj[:, 1], 0, frame.image.shape[0])
    x = frame.keypoints[frame.triangulated_idx]
    print(x[:, 0].shape, x[:, 1].shape)
    plt.imshow(frame.image)
    plt.scatter(x[:, 0], x[:, 1], marker="x", c="#1f77b4")
    plt.scatter(x_proj[:, 0], x_proj[:, 1], marker="x", c="#ff7f0e")
    plt.show()


if __name__ == "__main__":
    print("This is visualization script")
    points = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    RT1 = np.array(
        [
            [1.0000, 0.0000, 0.0000, 2.5000],
            [0.0000, 0.0000, -1.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.2500],
        ]
    )
    RT2 = np.array(
        [
            [1.0000, 0.0000, 0.0000, -2.5000],
            [0.0000, 0.0000, -1.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, -0.2500],
        ]
    )
    poses = [RT1]
    visualise_pose_and_3d_points(poses, points)

    path = "./data/stage1/box/estimated_points.ply"
    visualize_pointcloud(path)
