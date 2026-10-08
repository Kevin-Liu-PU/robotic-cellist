from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))
from analysis_paths import configured_path, require_directory, require_file
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


def sha256(filename: Path) -> str:
    hasher = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_pam_selection(protocol: dict) -> Path:
    """Require the configured PAM sources to match this run's path and hash pins.

    A lock without PAM source pins must be rebuilt with prepare.py before PAM
    scoring. Existing source files from another checkout do not authorize a
    newly selected checkout, even when their contents happen to match.
    """
    sources = {
        "pam_source": "PAM/PAM.py",
        "pam_config": "PAM/config.yml",
        "pam_dataset": "dataset.py",
    }
    models = protocol.get("models", {})
    for name in sources:
        entry = models.get(name, {})
        if not entry.get("path") or not entry.get("sha256"):
            raise ValueError(
                f"Missing PAM source pin: {name}. Set CELLO_PAM_ROOT and run "
                "prepare.py to create a fresh protocol lock."
            )
    pam_root = require_directory(configured_path("CELLO_PAM_ROOT"), "CELLO_PAM_ROOT")
    for name, relative in sources.items():
        entry = models[name]
        selected = (pam_root / relative).resolve()
        pinned = Path(entry["path"]).expanduser().resolve()
        if selected != pinned:
            raise ValueError(
                f"CELLO_PAM_ROOT differs from the protocol lock: {name}. "
                "Restore the locked source or run prepare.py to create a fresh lock."
            )
        require_file(selected, "CELLO_PAM_ROOT (locked PAM source)")
        if sha256(selected) != entry["sha256"]:
            raise ValueError(f"Model/interface changed after protocol lock: {name}")
    return pam_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-our", action="store_true")
    arguments = parser.parse_args()
    protocol = json.loads((ROOT / "PROTOCOL_LOCK.json").read_text())
    validate_pam_selection(protocol)
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for name, entry in protocol["models"].items():
        if entry["sha256"] and sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Model/interface changed after protocol lock: {name}")
    takes = json.loads((ROOT / "TAKES.json").read_text())
    selected_ids = list(takes) if arguments.all_our else protocol["interface_probe_take_ids"]
    selected = [takes[take_id] for take_id in selected_ids]
    analysis = "OUR_UNMATCHED_OVERVIEW" if arguments.all_our else "INTERFACE_PROBE_ONLY"
    destination = ROOT / ("PUBLIC_SCORES.json" if arguments.all_our else "PUBLIC_PROBE.json")
    for take in selected:
        if sha256(Path(take["wav"])) != take["wav_sha256"]:
            raise ValueError("Probe input changed since protocol lock")
    sys.path.insert(0, str(WORKSPACE / "external_judges"))
    import score_takes as existing

    output = {
        "purpose": "All90 original recordings, full-take unmatched overview only" if arguments.all_our else "Fixed first historical Base/RL pair, interface/runtime check only; not V1 or pitch-matched V2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": sha256(ROOT / "PROTOCOL_LOCK.json"),
        "runtime": sys.version, "threads": 2, "new_model_downloads_allowed": False,
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "torchaudio", "transformers", "numpy", "soundfile", "audiobox-aesthetics"]},
        "results": [], "errors": [],
    }

    def persist() -> None:
        destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    for model_name in ["Audiobox", "PAM"]:
        started = time.monotonic()
        try:
            if model_name == "Audiobox":
                model = existing.load_audiobox()
                model.model.to("cpu")
                model.device = torch.device("cpu")
            else:
                pam_root = validate_pam_selection(protocol)
                model, dataset_type = existing.load_pam(pam_root=pam_root)
            for take in selected:
                item_started = time.monotonic()
                try:
                    if model_name == "Audiobox":
                        values = {name: float(value) for name, value in model.forward([{"path": take["wav"]}])[0].items() if name in ["PQ", "PC"]}
                        expected = {"PQ", "PC"}
                    else:
                        dataset = dataset_type([take["wav"]])
                        _, audio, indices = dataset.collate([dataset[0]])
                        aggregate, chunks = model.evaluate(audio, indices)
                        values = {"PAM": float(aggregate[0])}
                        expected = {"PAM"}
                    if set(values) != expected or not all(np.isfinite(value) for value in values.values()):
                        raise ValueError(f"Invalid model output {values}")
                    row = {"take_id": take["take_id"], "segment_id": f"{take['take_id']}:full",
                           "wav_sha256": take["wav_sha256"], "model": model_name, "scores": values,
                           "seconds": time.monotonic() - item_started, "backend": "CPU",
                           "status": "ok", "analysis": analysis}
                    if model_name == "PAM":
                        row["chunks"] = [float(value) for value in chunks[0]]
                    output["results"].append(row)
                    print(json.dumps(row), flush=True)
                except Exception as error:
                    output["errors"].append({"model": model_name, "take_id": take["take_id"],
                                             "error_type": type(error).__name__, "message": str(error)})
                    print(f"FAILED {model_name} {take['take_id']}: {error}", flush=True)
                persist()
            del model
        except Exception as error:
            output["errors"].append({"model": model_name, "stage": "initialization",
                                     "error_type": type(error).__name__, "message": str(error)})
            print(f"UNAVAILABLE {model_name}: {error}", flush=True)
        output[f"{model_name}_total_seconds"] = time.monotonic() - started
        persist()


if __name__ == "__main__":
    main()
