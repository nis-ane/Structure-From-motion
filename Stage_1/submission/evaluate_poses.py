import numpy as np
import json


def compute_transformation_error(t1, t2):
    eps = 1e-6

    # Rotation
    r1 = t1[:3, :3]
    r2 = t2[:3, :3]
    rot_error = (np.trace(r1 @ r2.T) - 1) / 2
    rot_error = np.clip(rot_error, -1.0 + eps, 1.0 - eps)
    rot_error = np.arccos(rot_error)

    # Translation
    tr1 = t1[3, :3]
    tr2 = t2[3, :3]
    tr_error = np.linalg.norm(tr1 - tr2)

    return (rot_error, tr_error)


def pose_estimate(d1, d2):
    total_error_rotation = 0.0
    total_error_translation = 0.0

    keys = d1.keys()

    for camera in keys:
        transform1 = np.array(d1[camera], dtype=np.float32)
        transform2 = np.array(d2[camera], dtype=np.float32)

        rot_err, tr_err = compute_transformation_error(transform1, transform2)

        total_error_rotation += rot_err
        total_error_translation += tr_err

    total_error_rotation /= len(keys)
    total_error_translation /= len(keys)

    return total_error_rotation, total_error_translation


if __name__ == "__main__":
    f_gt = open("box/gt_camera_parameters.json", "r")
    f_predicted = open("box/estimated_camera_parameters.json", "r")

    camera_param_gt = json.load(f_gt)
    camera_param_predicted = json.load(f_predicted)

    rotation_error, translation_error = pose_estimate(
        camera_param_gt["extrinsics"], camera_param_predicted["extrinsics"]
    )

    print("Rotation error:", round(rotation_error, 2))
    print("Translation error:", round(translation_error, 3))
