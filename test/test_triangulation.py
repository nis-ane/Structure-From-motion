import numpy as np
from src.triangulation import triangulate_pts
from src.utils import project_3D_to_2D


def test_traingulation_using_crafted_data():
    P1 = np.array(
        [
            [700.0000, 320.0000, 0.0000, -95.0000],
            [0.0000, 240.0002, -933.3334, 173.3333],
            [0.0000, 1.0000, 0.0000, -0.2500],
        ]
    )
    P2 = np.array(
        [
            [700.0000, 320.0000, 0.0000, -255.0000],
            [0.0000, 240.0002, -933.3334, 173.3333],
            [0.0000, 1.0000, 0.0000, -0.2500],
        ]
    )

    X = np.array([[0, 5, 2, 1], [2, 5, 2, 1]])
    x1 = project_3D_to_2D(X, P1)
    x2 = project_3D_to_2D(X, P2)

    Xh = triangulate_pts(x1, x2, P1, P2)

    assert np.array_equal(np.round(X, 2), np.round(Xh, 2))
