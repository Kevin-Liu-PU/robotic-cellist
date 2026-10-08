from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECORD = "https://zenodo.org/records/10696680"
FILES = [
    "ReadMe_CelloSpringRecordings.txt",
    "MusicScore_Spring_CelloOpenStrings.pdf",
    "sound_player.wav",
    "sound_caseA.wav",
    "sound_caseB.wav",
    "DataMocap_Player.csv",
    "DataMocap_RobotCaseA.csv",
    "DataMocap_RobotCaseB.csv",
]


def digest(filename: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch(url: str, destination: Path, timeout: int, attempts: int) -> bool:
    if destination.exists() and destination.stat().st_size:
        print(f"PRESENT {destination.name}", flush=True)
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / "download_attempts.jsonl"
    previous = []
    if ledger.exists():
        previous = [json.loads(line) for line in ledger.read_text().splitlines()]
    spent = sum(entry["url"] == url for entry in previous)
    for attempt in range(spent + 1, attempts + 1):
        temporary = destination.with_suffix(destination.suffix + ".partial")
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        result = subprocess.run(
            ["curl", "-fL", "--connect-timeout", "15", "--max-time", str(timeout),
             "--silent", "--show-error", "--write-out", "%{http_code}",
             "--output", str(temporary), url],
            capture_output=True, text=True,
        )
        success = result.returncode == 0 and temporary.exists() and temporary.stat().st_size > 0
        entry = {
            "url": url, "destination": str(destination), "attempt": attempt,
            "started_utc": started, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "returncode": result.returncode, "http_status": result.stdout,
            "stderr": result.stderr, "success": success,
        }
        if success:
            temporary.replace(destination)
            destination.chmod(0o444)
            entry.update(size_bytes=destination.stat().st_size, sha256=digest(destination))
        with ledger.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(f"{'OK' if success else 'FAILED'} {destination.name} attempt={attempt} HTTP={result.stdout}", flush=True)
        if success:
            return True
        if attempt < attempts:
            time.sleep(2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--attempts", type=int, default=3, choices=(1, 2, 3))
    arguments = parser.parse_args()
    selected = arguments.only or FILES
    for name in selected:
        if name not in FILES:
            parser.error(f"Unrecognized public file: {name}")
    fetch("https://zenodo.org/api/records/10696680", ROOT / "sources/zenodo_record.json", 50, arguments.attempts)
    for name in selected:
        fetch(f"{RECORD}/files/{name}?download=1", ROOT / "raw" / name, 90, arguments.attempts)
    metadata_path = ROOT / "sources/zenodo_record.json"
    checks = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for entry in metadata.get("files", []):
            filename = ROOT / "raw" / entry["key"]
            if filename.exists() and entry.get("checksum"):
                algorithm, expected = entry["checksum"].split(":", 1)
                actual = digest(filename, algorithm)
                checks.append({"file": entry["key"], "algorithm": algorithm, "expected": expected,
                               "actual": actual, "match": actual == expected})
    (ROOT / "download_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    if any(not entry["match"] for entry in checks):
        raise SystemExit("Publisher checksum mismatch; do not analyze these inputs.")
    missing = [name for name in selected if not (ROOT / "raw" / name).exists()]
    print(json.dumps({"missing": missing, "verified_checksums": len(checks)}, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
