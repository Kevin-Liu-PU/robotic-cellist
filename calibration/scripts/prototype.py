from pathlib import Path
import numpy as np

from cello_bot.models.io import load_models
from cello_bot.geometry.transforms import pose6_to_matrix, transform_points
from cello_bot.calibration.cello_pose_cal import calibrate_cello_pose
from cello_bot.calibration.string_cal import fit_string_samples
from cello_bot.estimator.bow_state import BowStateEstimator
from cello_bot.planning.pose_generator import generate_tcp_pose

ROOT = Path(__file__).resolve().parents[1]

def main():
    bow, cello = load_models(ROOT / "configs" / "nominal.yaml")

    print("\\n=== CELLO BOT OFFLINE PROTOTYPE ===")
    print(f"Bow hair length: {bow.length:.3f} m")
    print(f"Strings loaded: {[s.id for s in cello.strings]}")
    print(f"Fiducials loaded: {list(cello.fiducials)}")

    # Synthetic known cello placement in robot base frame.
    angle = np.deg2rad(4.0)
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle),  np.cos(angle), 0],
                   [0, 0, 1]])
    T_true = np.eye(4)
    T_true[:3,:3] = Rz
    T_true[:3,3] = [0.42, -0.18, 0.31]

    measured = {
        fid: T_true[:3,:3] @ point + T_true[:3,3]
        for fid, point in cello.fiducials.items()
    }
    # Add tiny deterministic synthetic measurement noise.
    noise = {
        "F1": [0.0002, -0.0001, 0.0001],
        "F2": [-0.0001, 0.0002, 0.0],
        "F3": [0.0, -0.0001, 0.0002],
        "F4": [0.0001, 0.0, -0.0001],
    }
    measured = {k: v + np.array(noise[k]) for k, v in measured.items()}

    T_est, rms, residuals = calibrate_cello_pose(cello.fiducials, measured)
    print(f"\\nTier 1 calibration RMS: {rms*1000:.3f} mm")
    print(f"Recovered cello translation: {T_est[:3,3]} m")
    print(f"Fiducial residuals (mm): { {k: round(v*1000,3) for k,v in residuals.items()} }")

    # Synthetic string samples in cello coordinates, fit one line as a Tier-2 demo.
    a_string = next(s for s in cello.strings if s.id == "A")
    sample_points = np.array([
        a_string.bridge_pt + f*(a_string.far_pt-a_string.bridge_pt) + [0.0001, 0, 0.0001]
        for f in [0.0, 0.3, 0.6, 1.0]
    ])
    fit = fit_string_samples("A", sample_points)
    print(f"\\nTier 2 A-string line-fit RMS: {fit['rms']*1000:.3f} mm")
    print(f"Fitted tangent: {fit['tangent']}")

    # Create an illustrative tool pose in base frame. This is synthetic input,
    # not a hardware command or a validated playing configuration.
    # Put the hair segment near the A string in cello coordinates.
    T_base_tool = np.eye(4)
    T_base_tool[:3,:3] = T_est[:3,:3]
    hair_mid = 0.5*(bow.frog_tool + bow.tip_tool)
    target_cello = a_string.bridge_pt + 0.10*a_string.tangent
    T_base_tool[:3,3] = (T_est[:3,:3] @ (target_cello - hair_mid)
                         + T_est[:3,3])
    estimator = BowStateEstimator(bow, cello, T_est, contact_tolerance=0.01)
    state = estimator.estimate(T_base_tool)
    print("\\nBow-state estimate:")
    print(f"  closest string: {state.string_id}")
    print(f"  bow fraction:   {state.bow_fraction:.3f}")
    print(f"  string fraction:{state.string_fraction:.3f}")
    print(f"  segment gap:    {state.gap*1000:.2f} mm")
    print(f"  active:         {state.active}")

    # Demonstrate inverse geometric calculation without executing motion.
    target_pose = generate_tcp_pose(
        string_model=a_string,
        bow_model=bow,
        T_base_cello=T_est,
        distance_from_bridge=0.10,
        bow_fraction=0.50,
        clearance=0.0,
    )
    print("\\nIllustrative generated TCP pose [m, rad]:")
    print(np.array2string(target_pose, precision=5))
    print("\\nNOTE: offline calculation only; no robot connection or movement.")

if __name__ == "__main__":
    main()
