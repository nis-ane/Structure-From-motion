import numpy as np
from src.triangulation import triangulate_pts

def test_traingulation_using_crafted_data():
    P1 = np.array([[700.0000, 320.0000, 0.0000, -95.0000],[0.0000 ,240.0002, -933.3334, 173.3333],[0.0000, 1.0000, 0.0000, -0.2500]])
    P2 = np.array([[700.0000, 320.0000, 0.0000, -255.0000],[0.0000 ,240.0002, -933.3334, 173.3333],[0.0000, 1.0000, 0.0000, -0.2500]])
    
    X = np.array([[0,5,2,1],[2,5,2,1]])
    x1 = np.dot(P1,X.T)
    x1 = (x1/x1[-1]).T
    
    x2 = np.dot(P2,X.T)
    x2 = (x2/x2[-1]).T

    Xh = triangulate_pts(x1,x2,P1,P2)

    assert np.array_equal(np.round(X,2), np.round(Xh,2))
    