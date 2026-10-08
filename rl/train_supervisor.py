"""
rl/train_supervisor.py

Outer watchdog for train_piece_logged.py: if the training PROCESS dies for
a reason the in-process recovery cannot catch (a CoreAudio segfault, the
OOM killer, a python-level crash), relaunch it with --resume-run pointed at
the newest run directory and a --timesteps budget reduced by the strokes
already banked. Layers 1-2 (baseline_controller reconnect guard +
piece_env episode truncation) keep the process alive through robot
disconnects and protective stops; this layer exists for everything else.

    .venv/bin/python rl/train_supervisor.py [supervisor flags] -- \
        <every train_piece_logged.py argument, --timesteps N required>

e.g. the yunpiece retrain, supervised:

    CELLO_ALT_LOOP=1 caffeinate -dims .venv/bin/python rl/train_supervisor.py -- \
        yunpiece.mxl --real --timesteps 18200 --ckpt-every 10 \
        --no-calibrated-dynamics --reference-every 10

Behaviour:
  * child exit 0            -> done, supervisor exits 0
  * child crash             -> recount strokes across this session's run
                               dirs, relaunch with --resume-run <newest run
                               with episodes> --timesteps <remaining>
  * child HANGS: no stdout and no stroke-log growth for --hang-timeout
    seconds (measured 2026-09-01: a cable replug re-enumerated the USB
    NIC and stranded ur_rtde in a native reconnect that never raises and
    ignores SIGINT)     -> SIGKILL the child, then the same resume path;
                               --hang-timeout 0 disables the watchdog
  * two crashes in a row within --min-uptime seconds of launch
                            -> crash loop; STOP and wait for a human
                               (restarting into the same wall wastes robot
                               time and can grind the bow)
  * Ctrl-C                  -> forwarded to the child (which saves model +
                               replay buffer), then exit; NEVER restarts


"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = Path(os.environ.get("CELLO_TRAINER_OVERRIDE",
                              str(REPO_ROOT / "rl" / "train_piece_logged.py")))
RUN_DIR_RE = re.compile(r"^run dir -> (.+)$")


def strip_flag(args: list[str], flag: str, has_value: bool) -> list[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == flag:
            skip = has_value
            continue
        if has_value and a.startswith(flag + "="):
            continue
        out.append(a)
    return out


def get_flag_value(args: list[str], flag: str):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def set_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    out = strip_flag(args, flag, has_value=True)
    return out + [flag, value]


def count_run(run_dir: Path) -> tuple[int, int]:
    """(stroke records, episode records) in a run's stroke_log.jsonl.

    One env step writes exactly one stroke record, so the stroke count IS
    the timestep count the budget arithmetic needs. Reference episodes and
    the header are not steps and are excluded.
    """
    log = run_dir / "stroke_log.jsonl"
    strokes = episodes = 0
    if not log.exists():
        return 0, 0
    seen_eps = set()
    for line in open(log):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("header") or rec.get("reference"):
            continue
        if "episode" in rec and "stroke" in rec:
            strokes += 1
            seen_eps.add(rec["episode"])
    episodes = len(seen_eps)
    return strokes, episodes


def has_checkpoint(run_dir: Path) -> bool:
    """True if --resume-run can load something here: a latest.zip or any
    numbered ckpt. A restart that died before its first checkpoint leaves a
    run dir with episode records but nothing to resume from (2026-09-02
    16:34: picking that dir sent the next restart into a crash loop)."""
    return bool(list(run_dir.glob("sac_piece_*_latest.zip"))
                or list(run_dir.glob("ckpt_ep*.zip")))


def pick_resume_dir(run_dirs: list) -> "Path | None":
    """Newest run dir that has BOTH episode records and a checkpoint."""
    return next((d for d in reversed(run_dirs)
                 if count_run(d)[1] > 0 and has_checkpoint(d)), None)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="watchdog around train_piece_logged.py",
        usage="%(prog)s [--max-restarts N] [--min-uptime S] -- <trainer args>")
    ap.add_argument("--max-restarts", type=int, default=20)
    ap.add_argument("--hang-timeout", type=float, default=180.0,
                    help="SIGKILL the trainer when its stroke log has not "
                         "grown for this many seconds (stdout counts only "
                         "until the first stroke; reconnect chatter never "
                         "counts) -- a wedged native call never raises and "
                         "ignores SIGINT); the resume path then continues "
                         "the run. 0 disables. Longest legitimate silence "
                         "(PPO update + checkpoint save) is tens of "
                         "seconds, so 180 is a 10x margin.")
    ap.add_argument("--min-uptime", type=float, default=120.0,
                    help="a crash earlier than this after launch counts "
                         "toward the crash-loop guard (two in a row stop "
                         "the supervisor)")
    if "--" in sys.argv:
        split = sys.argv.index("--")
        own, fwd = sys.argv[1:split], sys.argv[split + 1:]
    else:
        own, fwd = [], sys.argv[1:]
    args = ap.parse_args(own)
    if not fwd:
        sys.exit("no trainer arguments given (put them after --)")

    target_s = get_flag_value(fwd, "--timesteps")
    if target_s is None:
        sys.exit("--timesteps N is required in the trainer arguments — the "
                 "supervisor needs the total budget to compute what remains "
                 "after a crash")
    target = int(target_s)

    run_dirs: list[Path] = []
    restarts = 0
    fast_crashes = 0
    interrupted = False

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        # the child receives its own SIGINT from the terminal's process
        # group; nothing to forward when run interactively, but cover the
        # detached case below at wait() time.

    signal.signal(signal.SIGINT, on_sigint)

    while True:
        done = sum(count_run(d)[0] for d in run_dirs)
        remaining = target - done
        if remaining <= 0:
            print(f"[supervisor] budget met: {done}/{target} strokes banked "
                  f"across {len(run_dirs)} run dir(s)")
            return

        cmd = [sys.executable, "-u", str(TRAINER)]
        if run_dirs:
            resume_from = pick_resume_dir(run_dirs)
            child_args = strip_flag(fwd, "--resume-run", True)
            child_args = strip_flag(child_args, "--resume", True)
            child_args = strip_flag(child_args, "--episode-offset", True)
            child_args = set_flag_value(child_args, "--timesteps",
                                        str(remaining))
            if resume_from is not None:
                child_args += ["--resume-run", str(resume_from)]
                print(f"[supervisor] restart {restarts}: resuming "
                      f"{resume_from.name}, {remaining}/{target} steps left")
            else:
                print(f"[supervisor] restart {restarts}: no run has episode "
                      f"records yet — relaunching fresh, full budget")
        else:
            child_args = list(fwd)
            print(f"[supervisor] launching: {TRAINER.name} "
                  + " ".join(child_args))
        cmd += child_args

        t_launch = time.time()
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 bufsize=1)
        last_out = time.time()
        hang_killed = False

        # Lines the reconnect guard prints every few seconds while the
        # robot is gone. They are not progress: on 2026-09-02 they kept
        # last_out fresh for 26 min while the trainer sat in a dead socket
        # and this watchdog never fired.
        CHATTER = ("[recover]", "RTDEControlInterface:", "Reconnecting",
                   "Broken pipe", "Lost connection to robot",
                   "control script is not running")

        def pump():
            nonlocal last_out
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()          # fix 3: the log file used to stay empty for minutes
                if not any(tok in line for tok in CHATTER):
                    last_out = time.time()
                m = RUN_DIR_RE.match(line.rstrip())
                if m:
                    d = Path(m.group(1))
                    if d not in run_dirs:
                        run_dirs.append(d)

        def last_activity() -> float:
            """Newest sign of life. Before the first stroke is logged
            (model load, robot connect, first episode) stdout is the only
            signal. Once the stroke log exists, ONLY its growth counts:
            a trainer that talks but does not play is hung."""
            if run_dirs:
                try:
                    return (run_dirs[-1] / "stroke_log.jsonl").stat().st_mtime
                except OSError:
                    pass
            return last_out

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        while child.poll() is None:
            time.sleep(1.0)
            if interrupted:
                child.send_signal(signal.SIGINT)
                break
            if args.hang_timeout > 0:
                quiet = time.time() - last_activity()
                if quiet > args.hang_timeout:
                    hang_killed = True
                    print(f"[supervisor] HANG: no stdout and no stroke-log "
                          f"growth for {quiet:.0f}s "
                          f"(--hang-timeout {args.hang_timeout:.0f}) — "
                          f"SIGKILL; a wedged native call never raises and "
                          f"ignores SIGINT, so killing is the only exit")
                    child.kill()
                    break
        if interrupted and child.poll() is None:
            try:
                rc = child.wait(timeout=120.0)
            except subprocess.TimeoutExpired:
                print("[supervisor] child ignored SIGINT for 120s (wedged "
                      "native call?) — SIGKILL")
                child.kill()
                rc = child.wait()
        else:
            rc = child.wait()
        reader.join(timeout=5.0)
        uptime = time.time() - t_launch

        if interrupted:
            print(f"[supervisor] Ctrl-C: child exited {rc} after "
                  f"{uptime:.0f}s; NOT restarting. Resume by re-running the "
                  f"same supervisor command.")
            return
        if rc == 0:
            done = sum(count_run(d)[0] for d in run_dirs)
            print(f"[supervisor] trainer finished cleanly "
                  f"({done}/{target} strokes across {len(run_dirs)} run "
                  f"dir(s))")
            return

        fast_crashes = fast_crashes + 1 if uptime < args.min_uptime else 0
        restarts += 1
        how = ("killed by hang watchdog" if hang_killed
               else f"died (exit {rc})")
        print(f"[supervisor] trainer {how} after {uptime:.0f}s "
              f"— restart {restarts}/{args.max_restarts}")
        if fast_crashes >= 2:
            sys.exit("[supervisor] CRASH LOOP: two crashes within "
                     f"{args.min_uptime:.0f}s of launch. Something is broken "
                     "that a restart will not fix (bad flag, dead audio "
                     "device, robot in a state recovery cannot clear). "
                     "Stopping for a human.")
        if restarts > args.max_restarts:
            sys.exit(f"[supervisor] gave up after {args.max_restarts} "
                     f"restarts")
        time.sleep(3.0)


if __name__ == "__main__":
    main()
