#!/usr/bin/env zsh
set -eu
ROOT=${0:A:h:h}
[[ -d "$ROOT/rl" && -f "$ROOT/rl/g1_session.py" ]] || { print -u2 -- "Repository entry point missing: $ROOT/rl/g1_session.py"; exit 2; }
# CELLO_PYTHON is an absolute executable path; G1_PYTHON remains a compatibility alias.
PY=${CELLO_PYTHON:-${G1_PYTHON:-$ROOT/.venv/bin/python}}
[[ "$PY" == /* && -f "$PY" && -x "$PY" ]] || { print -u2 -- "Set CELLO_PYTHON to an existing absolute Python executable, or prepare $ROOT/.venv/bin/python."; exit 2; }
cd -- "$ROOT" || exit 2
exec "$PY" "$ROOT/rl/g1_session.py" "$@"
