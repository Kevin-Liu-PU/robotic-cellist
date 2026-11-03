# Cello Bot – Offline Prototype

This prototype implements the **mathematical/software core** of the cello-bowing architecture:
- UR-style pose (XYZ + rotation vector) and homogeneous-transform conversion
- Bow and cello geometry models loaded from YAML
- Rigid alignment (Kabsch) for cello-pose calibration
- 3D line fitting for measured string samples
- Bow-to-string closest-point estimation
- Inverse geometric target calculation (returns a pose; does not move a robot)
- Synthetic end-to-end demo and unit tests

## Safety and scope

This is an offline prototype. It does **not** connect to `ur_rtde`, send robot commands,
probe physical objects, or provide hardware motion parameters. Do not use calculated poses
as motion commands without a separate, professionally reviewed safety and integration process.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

## Run the prototype

From the project root:

```bash
python scripts/prototype.py
```

Run tests:

```bash
pytest
```

## Coordinate conventions

- Distances: metres.
- Angles: radians.
- A transform `T_a_b` maps coordinates expressed in frame `b` into frame `a`.
- Frames: `base`, `tool`, and `cello`.
- Cello frame: example convention is X across the strings, Y from bridge toward fingerboard,
  and Z away from the cello top.
- String IDs in the example are `C`, `G`, `D`, `A`.

The example YAML values are **synthetic placeholders**, not measurements of your instrument.
Replace them with measured/CAD values before treating outputs as representative.

## Important modelling note

A string is represented as a finite line segment. The bow hair is also represented as a
finite line segment. The estimator reports the closest geometric points and separation.
It is a prototype geometric estimator, not a validated contact/force detector. The simple
orientation metrics are diagnostic approximations.
