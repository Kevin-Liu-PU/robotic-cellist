"""Local path configuration shared by offline analysis scripts.

CELLO_REPO_ROOT defaults to this checkout. CELLO_DATA_ROOT defaults to its
rl/checkpoints_piece directory. Other roots have no external-machine defaults.
Environment paths may be absolute or relative to the caller's working directory.
This module reads no experiment files and imports no model or hardware libraries.
"""

import os
from pathlib import Path


def configured_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value is not None and not value.strip():
        raise ValueError(f"{name} is empty; set a local path or unset it.")
    path = Path(value).expanduser() if value is not None else default
    return path.resolve() if path is not None else None


def repo_root() -> Path:
    return configured_path("CELLO_REPO_ROOT", Path(__file__).resolve().parents[1])


def data_root() -> Path:
    return configured_path("CELLO_DATA_ROOT", repo_root() / "rl" / "checkpoints_piece")


def require_file(path: Path | str | None, setting: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"Set {setting} to an authorized local file; private inputs are not distributed.")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing input file: {path}. Check {setting}; private inputs are not distributed.")
    return path


def require_directory(path: Path | str | None, setting: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"Set {setting} to an authorized local directory.")
    path = Path(path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing input directory: {path}. Check {setting}.")
    return path


def recorded_take_path(value: str, root: Path) -> Path:
    """Relocate a logged perform_* directory by ID, accepting either OS separator.

The take ID is preserved; a former machine's absolute prefix is not followed.
Pair selection and labels remain the caller's responsibility.
"""
    name = value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not name.startswith("perform_") or name in {".", ".."}:
        raise ValueError("Expected a logged perform_* take directory under CELLO_DATA_ROOT.")
    return root / name
