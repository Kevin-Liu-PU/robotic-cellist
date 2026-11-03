"""Offline inverse geometric pose calculation; deliberately does not command a robot."""
import numpy as np
from scipy.spatial.transform import Rotation
from cello_bot.geometry.transforms import matrix_to_pose6, validate_transform

def generate_tcp_pose(
    string_model,
    bow_model,
    T_base_cello,
    T_tool_hair_rotation=None,
    distance_from_bridge=0.10,
    bow_fraction=0.50,
    clearance=0.0,
):
    """
    Calculate an illustrative TCP pose for a point on a cello string.

    T_tool_hair_rotation is the fixed rotation mapping tool-frame vectors to the
    desired cello-frame bow orientation. If omitted, identity is used as a
    mathematical placeholder. The result is a pose only, NOT a validated or
    safe robot motion target.
    """
    T_base_cello = validate_transform(T_base_cello)
    if not 0.0 <= bow_fraction <= 1.0:
        raise ValueError("bow_fraction must be between 0 and 1")
    if not 0.0 <= distance_from_bridge <= np.linalg.norm(string_model.far_pt-string_model.bridge_pt):
        raise ValueError("distance_from_bridge lies outside the modelled string segment")
    hair_rotation = np.eye(3) if T_tool_hair_rotation is None else np.asarray(T_tool_hair_rotation, dtype=float)
    if hair_rotation.shape != (3,3):
        raise ValueError("T_tool_hair_rotation must be 3x3")
    # Target contact point in cello frame.
    contact = string_model.bridge_pt + distance_from_bridge * string_model.tangent
    contact = contact - clearance * string_model.normal
    # The bow hair axis should span across the string, not run along it.
    # Use b = normal x string_tangent as the nominal bow-axis direction.
    # This is a basic geometric placeholder; real bowing orientation needs setup-specific calibration.
    desired_hair_axis = np.cross(string_model.normal, string_model.tangent)
    axis_norm = np.linalg.norm(desired_hair_axis)
    if axis_norm < 1e-9:
        raise ValueError("String normal and tangent do not define a bow axis")
    desired_hair_axis = desired_hair_axis / axis_norm
    tool_hair_axis = (bow_model.tip_tool-bow_model.frog_tool)/bow_model.length
    # Build minimal rotation from tool hair axis to desired bow axis.
    cross = np.cross(tool_hair_axis, desired_hair_axis)
    dot = float(np.clip(tool_hair_axis @ desired_hair_axis, -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-10:
        R_align = np.eye(3) if dot > 0 else Rotation.from_rotvec(np.pi*np.array([1.,0.,0.])).as_matrix()
    else:
        axis = cross/np.linalg.norm(cross)
        angle = np.arccos(dot)
        R_align = Rotation.from_rotvec(axis*angle).as_matrix()
    R_tool_cello = R_align @ hair_rotation
    # TCP origin such that the selected point on the hair maps to contact.
    hair_local = bow_model.frog_tool + bow_fraction*(bow_model.tip_tool-bow_model.frog_tool)
    p_tcp_cello = contact - R_tool_cello @ hair_local
    R_base_cello = T_base_cello[:3,:3]
    p_tcp_base = R_base_cello @ p_tcp_cello + T_base_cello[:3,3]
    R_base_tool = R_base_cello @ R_tool_cello
    T_base_tool = np.eye(4)
    T_base_tool[:3,:3] = R_base_tool
    T_base_tool[:3,3] = p_tcp_base
    return matrix_to_pose6(T_base_tool)
