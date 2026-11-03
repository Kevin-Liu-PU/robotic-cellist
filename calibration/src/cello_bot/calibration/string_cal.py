"""Offline string-line fitting from already-collected 3D sample points."""
import numpy as np
from cello_bot.geometry.fitting import fit_line

def fit_string_samples(string_id: str, measured_points_cello):
    """
    Fit a line to measured points already expressed in cello coordinates.
    Returns a dict with centroid, unit tangent, residuals, and sample span.
    This function does not collect points or touch hardware.
    """
    center, direction, residuals = fit_line(measured_points_cello)
    P = np.asarray(measured_points_cello, dtype=float)
    along = (P-center) @ direction
    p0 = center + along.min()*direction
    p1 = center + along.max()*direction
    return {
        "id": string_id,
        "point": center,
        "tangent": direction,
        "sample_start": p0,
        "sample_end": p1,
        "rms": float(np.sqrt(np.mean(residuals**2))),
        "residuals": residuals,
    }
