"""Closest points between finite 3D line segments."""
from __future__ import annotations
import numpy as np

def closest_points_segments(p1, q1, p2, q2):
    """
    Return (point_on_segment_1, point_on_segment_2, u, v, distance).
    u and v are normalized segment parameters in [0,1].
    """
    p1, q1, p2, q2 = [np.asarray(x, dtype=float) for x in (p1, q1, p2, q2)]
    d1, d2, r = q1-p1, q2-p2, p1-p2
    a, e = d1@d1, d2@d2
    eps = 1e-12
    if a < eps or e < eps:
        raise ValueError("Segments must have non-zero length")
    b, c, f = d1@d2, d1@r, d2@r
    denom = a*e - b*b
    if denom > eps:
        u = np.clip((b*f - c*e)/denom, 0.0, 1.0)
    else:
        u = 0.0
    v = (b*u + f)/e
    if v < 0.0:
        v = 0.0
        u = np.clip(-c/a, 0.0, 1.0)
    elif v > 1.0:
        v = 1.0
        u = np.clip((b-c)/a, 0.0, 1.0)
    cp1, cp2 = p1 + u*d1, p2 + v*d2
    return cp1, cp2, float(u), float(v), float(np.linalg.norm(cp1-cp2))
