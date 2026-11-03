"""Offline Tier-1 cello pose calibration from corresponding fiducials."""
import numpy as np
from cello_bot.geometry.fitting import kabsch

def calibrate_cello_pose(nominal_fiducials: dict, measured_base_points: dict):
    """
    Estimate T_base_cello from matching fiducial dictionaries.
    Inputs map fiducial IDs to 3D points.
    Returns (T_base_cello, rms, residual_by_id).
    """
    ids = sorted(set(nominal_fiducials) & set(measured_base_points))
    if len(ids) < 3:
        raise ValueError("At least 3 matching fiducials are required")
    if set(nominal_fiducials) != set(measured_base_points):
        raise ValueError("Nominal and measured fiducial IDs must match exactly")
    model = np.array([nominal_fiducials[k] for k in ids], dtype=float)
    measured = np.array([measured_base_points[k] for k in ids], dtype=float)
    T, rms, residuals = kabsch(model, measured)
    return T, rms, {k: float(r) for k, r in zip(ids, residuals)}
