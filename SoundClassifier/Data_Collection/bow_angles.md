# Bow-angle calibration anchors

This note describes a three-angle calibration design: bottom, reference and top,
with a frog and tip pose for each angle. These anchors describe a proposed
collection configuration. They do not establish collision avoidance, safe contact
forces, or a policy that learns to select angles between the anchors.

The historical names `BOTTOM_ANGLE_*`, `REFERENCE_ANGLE_*`, `TOP_ANGLE_*` and
`EXPERIMENT_MODE` are not configuration variables in the current
`recording_script.py`. Do not add them and assume that the collector will use
them. That script currently builds its systematic grid from `DEPTH_LEVELS`,
`SPEED_LEVELS` and `BOW_ZONES`. Review the actual collector and controller
configuration before preparing a hardware session.

## The three anchors

For the design described here:

1. **Bottom angle** is one reviewed wrist orientation.
2. **Reference angle** is the chosen neutral orientation for the target string.
3. **Top angle** is the other reviewed wrist orientation.

Each anchor needs two TCP poses: one at the frog end and one at the tip end of
the intended stroke. A pose is `[x, y, z, rx, ry, rz]`: position in metres and a
rotation vector in radians. The orientation values are not Euler angles.

These poses depend on the robot installation, tool, bow and instrument placement.
Do not reuse example coordinates or a previous installation's poses as motion
targets. Reaching two endpoints does not establish that the interpolated path,
contact force or reset motion between them is acceptable.

## Record poses with the teach pendant

An operator must first review the tool and base frames, reach and joint limits,
clearance, contact limits, reset procedure and emergency stop for the setup.
Use the lab's attended calibration procedure when moving the robot.

For each reviewed angle:

1. Position the bow at the intended frog endpoint using the teach pendant.
2. Read and record the TCP pose and the active tool/base-frame configuration.
3. Position the bow at the intended tip endpoint and record its TCP pose.
4. For a fixed-orientation stroke, keep the wrist orientation consistent between
   the two endpoints. Verify the path and clearance separately.

Repeat for the reference and other extreme orientation. A previous zone
experiment's reference is usable only after checking it against the current
installation; it is not automatically a valid calibration.

## Record the calibration design

The following names illustrate the six required records. They are design labels,
not a runnable configuration block for the current collector:

```text
BOTTOM_ANGLE_FROG    = [x, y, z, rx, ry, rz]
BOTTOM_ANGLE_TIP     = [x, y, z, rx, ry, rz]
REFERENCE_ANGLE_FROG = [x, y, z, rx, ry, rz]
REFERENCE_ANGLE_TIP  = [x, y, z, rx, ry, rz]
TOP_ANGLE_FROG       = [x, y, z, rx, ry, rz]
TOP_ANGLE_TIP        = [x, y, z, rx, ry, rz]
```

Record the measured values, coordinate frames and calibration conditions in the
private session records. Connect reviewed poses to the actual collector's
configuration only after verifying how it uses them. This note does not authorize
a hardware run or validate the collection code.

## Planned collection size

A proposed three-angle grid with five depths and five speeds contains
`3 * 5 * 5 = 75` conditions. Five repetitions would request `75 * 5 = 375`
recordings, before any separately specified robustness conditions. These are
planning counts, not evidence that recordings were collected or accepted.

The current `recording_script.py` has five `DEPTH_LEVELS`, five `SPEED_LEVELS`
and `N_REPEATS = 5`, but varies bow zones rather than the three angle anchors
above. Its actual condition count also includes `BAD_CONDITIONS`. Determine
the planned size from the selected script's configuration; do not infer a
three-angle session from the repetition count alone.

## Session records and validation

Keep commanded settings separate from measured state. For a future angle-based
collection, record the angle identity, both endpoint poses, commanded depth and
speed, and the measured motion and sensor streams needed to assess the stroke.
Confirm the emitted metadata schema in the collector; this note does not promise
`commanded.angle_name` or `commanded.angle_t` fields in `recording_script.py`.

Inspect completed records for missing signals, interrupted strokes, coordinate
consistency and repeatability. A torque-derived estimate is not automatically a
calibrated contact-force measurement. Fixed calibration anchors can document a
collection procedure, but they do not by themselves establish its safety,
repeatability or learned cross-angle adaptation. Keep collected recordings,
sensor traces and session logs outside the public repository.
