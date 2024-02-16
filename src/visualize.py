import trimesh
import numpy as np


def get_poses_obj(poses, size=0.1):
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
        segs = trimesh.load_path(segs)
        objects.append(segs)
    return objects


def visualise_pose_and_3d_points(poses, pointcloud, colors=None):
    poses_obj = get_poses_obj(poses)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=pointcloud, colors=colors))
    scene.add_geometry(poses_obj)
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.show()


def visualise_3d_points(points_3d):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points_3d))
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.show()


def visualise_poses(poses):
    poses_obj = get_poses_obj(poses)
    scene = trimesh.Scene()
    scene.add_geometry(poses_obj)
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.show()


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
