import trimesh
import numpy as np
from src.utils import get_correspondence_from_file

view_angle = [0, 0, np.pi]
distance = 5
cam_center = [0, 0, 5]


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
    scene.set_camera(angles=view_angle, distance=distance, center=cam_center)
    scene.show()


def visualise_3d_points(points_3d, colors=None):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points_3d, colors=colors))
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.set_camera(angles=view_angle, distance=distance, center=cam_center)
    scene.show()


def visualise_poses(poses):
    poses_obj = get_poses_obj(poses)
    scene = trimesh.Scene()
    scene.add_geometry(poses_obj)
    scene.add_geometry(trimesh.creation.axis(axis_length=0.4))
    scene.set_camera(angles=view_angle, distance=distance, center=cam_center)
    scene.show()


if __name__ == "__main__":
    print("This is visualization script")
    # Load 2D correspondences
    correspondences_file = '/Users/hewanshrestha/Desktop/3dcv_project/Stage_1/submission/box/correspondences/0_2.txt'
    points1, points2 = get_correspondence_from_file(correspondences_file)
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
    R1_t = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    # R1_t = np.array(
    #     [
    #         [ 9.98886516e-01, -1.96813011e-03,  4.71365621e-02, -6.30377419e-01],
    #         [ 9.55236439e-05,  9.99211775e-01,  3.96966052e-02, -2.38625753e-01],
    #         [-4.71775359e-02, -3.96479010e-02,  9.98099356e-01,  7.38702958e-01],
    #     ]
    # )
    R2_t = np.array(
        [
            [ 0.99827172, -0.01316255, -0.05727411, 0.62592621],
            [ 0.01146899,  0.99949013, -0.02979826, 0.23789855],
            [ 0.05763713,  0.02908988,  0.99791369, -0.74271169],
        ]
    )

    # [0.9749113949459054, 0.0037593517902943276, -0.22255913745180414, 0.5958810592522974], 
    # [-0.022029286579879123, 0.9965772107711804, -0.07966771678846588, 0.18467123463507917],
    # [0.2214976778052964, 0.08257219376961197, 0.9716586145577645, 0.4665736593665555],
    
    poses = [R1_t, R2_t]
    visualise_pose_and_3d_points(poses, points1)
