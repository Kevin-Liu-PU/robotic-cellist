"""Pure coordinate-transform utilities. No robot/hardware dependencies."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

def pose6_to_matrix(pose6) -> np.ndarray:
    """Convert UR-style [x,y,z,rx,ry,rz] to a 4x4 transform."""
    p = np.asarray(pose6, dtype=float)
    if p.shape != (6,):
        raise ValueError("pose6 must contain exactly 6 values")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix()
    T[:3, 3] = p[:3]
    return T

def matrix_to_pose6(T) -> np.ndarray:
    """Convert a 4x4 transform to UR-style [x,y,z,rx,ry,rz]."""
    T = validate_transform(T)
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]

def validate_transform(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("Transform must be 4x4")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Invalid homogeneous transform bottom row")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Rotation block is not orthonormal")
    if np.linalg.det(T[:3, :3]) < 0.999:
        raise ValueError("Rotation block must have determinant +1")
    return T

def invert_transform(T) -> np.ndarray:
    T = validate_transform(T)
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out

def transform_points(T, points) -> np.ndarray:
    """Transform one point (3,) or an array of points (N,3)."""
    T = validate_transform(T)
    pts = np.asarray(points, dtype=float)
    if pts.shape == (3,):
        return T[:3, :3] @ pts + T[:3, 3]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (3,) or (N,3)")
    return (T[:3, :3] @ pts.T).T + T[:3, 3]

def transform_direction(T, direction) -> np.ndarray:
    """Transform a direction vector (rotation only), then normalize it."""
    T = validate_transform(T)
    v = T[:3, :3] @ np.asarray(direction, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Direction vector cannot be zero")
    return v / n
