import numpy as np


def Vec2Skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def project_3D_to_2D(X, P):
    x = np.dot(P, X.T)
    x = (x / x[-1]).T
    return x


def get_correspondence_from_file(file):
    f = open(file, "r")
    pts_1 = []
    pts_2 = []
    for x in f:
        correspondence = [float(coor) for coor in x.split(" ")]
        pts_1.append((correspondence[0], correspondence[1], 1))
        pts_2.append((correspondence[2], correspondence[3], 1))
    return np.array(pts_1, dtype=np.float32), np.array(pts_2, dtype=np.float32)


def quaternion_to_rotation_matrix(q):
    """
    Convert a unit quaternion to a 3x3 rotation matrix.

    Parameters:
        q (numpy.ndarray): Unit quaternion [w, x, y, z]

    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    # Ensure that the input quaternion is valid
    assert len(q) == 4, "Input quaternion must have 4 elements"
    norm = np.linalg.norm(q)
    assert np.isclose(norm, 1.0), "Input quaternion must be unit quaternion"

    x, y, z, w = q
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w

    rotation_matrix = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)],
        ]
    )

    return rotation_matrix


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a unit quaternion.

    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix

    Returns:
        numpy.ndarray: Unit quaternion [w, x, y, z]
    """
    # Ensure that the input matrix is a valid rotation matrix
    assert R.shape == (3, 3), "Input matrix must be 3x3"
    assert np.isclose(
        np.linalg.det(R), 1.0
    ), "Input matrix must have determinant 1 (be a rotation matrix)"

    # Compute the quaternion elements
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w])
