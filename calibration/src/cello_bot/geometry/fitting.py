"""Least-squares geometry fitting utilities."""
from __future__ import annotations
import numpy as np

def kabsch(model_points, measured_points):
    """
    Find T_model_to_measured such that measured ~= R @ model + t.
    Returns (4x4 transform, RMS residual in metres, per-point residuals).
    At least 3 non-collinear corresponding points are required.
    """
    A = np.asarray(model_points, dtype=float)
    B = np.asarray(measured_points, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("Point arrays must both have shape (N,3)")
    if len(A) < 3:
        raise ValueError("At least 3 corresponding points are required")
    if np.linalg.matrix_rank(A - A.mean(axis=0)) < 2:
        raise ValueError("Model points must not be collinear")
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    predicted = (R @ A.T).T + t
    residuals = np.linalg.norm(predicted - B, axis=1)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T, float(np.sqrt(np.mean(residuals**2))), residuals

def fit_line(points):
    """
    PCA least-squares line fit.
    Returns (centroid_on_line, unit_direction, perpendicular_residuals).
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        raise ValueError("At least two 3D points are required")
    center = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - center, full_matrices=False)
    direction = Vt[0]
    # Deterministic sign: point toward positive Y where possible.
    if direction[1] < 0:
        direction = -direction
    delta = P - center
    projection = np.outer(delta @ direction, direction)
    residuals = np.linalg.norm(delta - projection, axis=1)
    return center, direction / np.linalg.norm(direction), residuals
