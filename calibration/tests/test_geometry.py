import numpy as np
from scipy.spatial.transform import Rotation
from cello_bot.geometry.transforms import pose6_to_matrix, matrix_to_pose6, transform_points
from cello_bot.geometry.fitting import kabsch, fit_line
from cello_bot.geometry.segments import closest_points_segments

def test_pose_round_trip():
    p = np.array([0.2, -0.1, 0.4, 0.1, -0.2, 0.3])
    out = matrix_to_pose6(pose6_to_matrix(p))
    assert np.allclose(p[:3], out[:3])
    assert np.allclose(pose6_to_matrix(p), pose6_to_matrix(out))

def test_kabsch_recovers_transform():
    A = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    R = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
    t = np.array([0.4, -0.2, 0.1])
    B = (R @ A.T).T + t
    T, rms, residuals = kabsch(A, B)
    assert np.allclose(T[:3,:3], R)
    assert np.allclose(T[:3,3], t)
    assert rms < 1e-10
    assert np.max(residuals) < 1e-10

def test_fit_line():
    P = np.array([[0,0,0],[0,1,0],[0,2,0]], dtype=float)
    point, direction, residuals = fit_line(P)
    assert abs(abs(direction[1])-1) < 1e-8
    assert np.max(residuals) < 1e-8

def test_segment_distance():
    p1, p2, u, v, d = closest_points_segments(
        [0,0,0], [1,0,0], [0.5,1,0], [0.5,1,1]
    )
    assert np.allclose(p1, [0.5,0,0])
    assert np.allclose(p2, [0.5,1,0])
    assert abs(d-1.0) < 1e-8
