"""
score_takes.py -- external pretrained scoring baselines for recorded takes.

Two pretrained scoring models:
  Audiobox Aesthetics (Meta, 2025): CE content enjoyment, CU content
      usefulness, PC production complexity, PQ production quality; 1..10.
      16 kHz mono, 10 s windows, weighted mean over windows.
  PAM (Deshmukh et al., Interspeech 2024): CLAP antonym prompting,
      P("clear and clean" vs "noisy and with artifacts"), 0..1.
      44.1 kHz, 7 s chunks, mean over chunks (short tail = last 7 s).

Full-take level only (notes are far below both models' windows).
Manifest: A/B logs in rl/checkpoints_piece (=== LABEL / perform -> DIR).

Usage (judge venv):
  /path/to/judge-venv/bin/python score_takes.py [--only COND,...]
Set CELLO_DATA_ROOT to the local directory containing the historical A/B logs
and perform_* takes; it defaults to this checkout's rl/checkpoints_piece.
CELLO_REPO_ROOT overrides the source checkout. CELLO_PAM_ROOT explicitly selects
a local PAM checkout when scoring. --help requires no model libraries or data.
Writes results/scores_<date>.json and prints paired-delta tables.

Revision: 2026-09-02.
"""
import sys, re, json, glob, time, argparse, statistics as st
from pathlib import Path
from collections.abc import Mapping
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_paths import configured_path, repo_root, data_root, require_file, require_directory, recorded_take_path

REPO = repo_root()
CP = data_root()
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

# ---------- manifest ----------
def parse_log(name):
    """[(label, dir)] from an A/B or bracket log."""
    out, label = [], None
    log_path = require_file(CP / name, "CELLO_DATA_ROOT (A/B logs)")
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"=== (\S+) \d\d:\d\d:\d\d", line)
        if m: label = m.group(1); continue
        m = re.search(r"perform -> (.+)", line)
        if m and label: out.append((label, recorded_take_path(m.group(1).strip(), CP))); label = None
    return out

def pairs_from(takes, base_prefix="Base", rl_prefix="RL"):
    """Back-to-back (Base_k, RL_k) pairs by index k."""
    b = {l[len(base_prefix):]: d for l, d in takes if l.startswith(base_prefix)}
    r = {l[len(rl_prefix):]: d for l, d in takes if l.startswith(rl_prefix)}
    return [(b[k], r[k]) for k in sorted(b) if k in r]

P = lambda s: CP / f"perform_{s}"
def build_manifest():
    """Resolve the unchanged historical pair index only when analysis requests it."""
    manifest = {
      # condition: {"pairs": [(Base, RL), ...], "note": ...}
      "twinkle_alt2_s0 (9/1 20:31, Base vs ALT2 final)": {"pairs": [
          (P("20260901_203157"), P("20260901_203332")), (P("20260901_203418"), P("20260901_203552")),
          (P("20260901_203638"), P("20260901_203813"))]},
      "twinkle_alt_ep40 (9/1 20:31, same Base vs ALT ep40)": {"pairs": [
          (P("20260901_203157"), P("20260901_203244")), (P("20260901_203418"), P("20260901_203505")),
          (P("20260901_203638"), P("20260901_203725"))]},
      "twinkle_alt_1830 (9/1 18:31, Base vs ALT ep40)": {"pairs": pairs_from(parse_log("ab_twinkle_alt_1830.log"))},
      "twinkle_alt2_s1 (9/1 22:44)": {"pairs": pairs_from(parse_log("ab_twinkle_alt2_s1.log"))},
      "twinkle_alt2_s2 (9/2 00:11)": {"pairs": pairs_from(parse_log("ab_twinkle_alt2_s2.log"))},
      "twinkle_a11 (9/2 01:02, Base vs A11 final)": {"pairs": pairs_from(parse_log("ab_twinkle_a11_20260902.log"))},
      "popper_alt2_s0 (9/2 05:37)": {"pairs": pairs_from(parse_log("ab_popper_alt2_s0.log"))},
      "bracket_after_twinkle_s2 (9/2 00:17, s0 final vs B1/B2)": {"pairs": None, "log": "bracket_twinkle_0902_after_twinkle_s2.log"},
      "bracket_before_popper (9/2 01:06)": {"pairs": None, "log": "bracket_twinkle_0902_before_popper.log"},
      "bracket_after_popper_s0 (9/2 06:01)": {"pairs": None, "log": "bracket_twinkle_after_popper_s0.log"},
      "a8_compiled_vs_live (9/2 00:20, s0 final; compiled=A, live=B)": {"pairs": [
          (P("20260902_002026"), P("20260902_002228")), (P("20260902_002311"), P("20260902_002512")),
          (P("20260902_002125"), P("20260902_002228")), (P("20260902_002410"), P("20260902_002512"))]},
    }
    for k, v in manifest.items():
        if v["pairs"] is None:  # bracket: B1 -> POL -> B2 ; pair POL with each Base
            t = dict(parse_log(v["log"]))
            v["pairs"] = [(t["B1"], t["POL"]), (t["B2"], t["POL"])]
    return manifest


class _LazyManifest(Mapping):
    def __init__(self):
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = build_manifest()
        return self._data

    def __getitem__(self, key):
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


MANIFEST = _LazyManifest()

# ---------- judges ----------
def sf_load(path, *a, **k):
    import torch
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(x.T.copy()), sr

def load_audiobox():
    import audiobox_aesthetics.infer as inf
    inf.torchaudio.load = sf_load
    return inf.initialize_predictor()

def validate_pam_import_sources(pam_root):
    """Reject another checkout's cached PAM/dataset modules without replacing them."""
    package_root = (pam_root / "PAM").resolve()
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name == "dataset":
            filename = getattr(module, "__file__", None)
            valid = filename is not None and Path(filename).resolve() == (pam_root / "dataset.py").resolve()
        elif name == "PAM" or name.startswith("PAM."):
            filename = getattr(module, "__file__", None)
            if filename is not None:
                valid = Path(filename).resolve().is_relative_to(package_root)
            else:
                locations = list(getattr(module, "__path__", []))
                valid = bool(locations) and all(Path(p).resolve().is_relative_to(package_root) for p in locations)
        else:
            continue
        if not valid:
            raise RuntimeError(f"Cached {name} module is outside the selected PAM checkout; use a fresh Python process.")


def load_pam(pam_root=None):
    pam_root = require_directory(pam_root if pam_root is not None else configured_path("CELLO_PAM_ROOT"),
                                 "CELLO_PAM_ROOT (local PAM checkout)")
    require_file(pam_root / "PAM" / "PAM.py", "CELLO_PAM_ROOT")
    require_file(pam_root / "PAM" / "config.yml", "CELLO_PAM_ROOT")
    require_file(pam_root / "dataset.py", "CELLO_PAM_ROOT")
    validate_pam_import_sources(pam_root)
    import torch
    import soundfile as sf
    import torchaudio
    torchaudio.load = sf_load
    sys.path.insert(0, str(pam_root))
    from PAM import PAM
    class PAMFixed(PAM):  # transformers>=4.4x: encode_plus removed
        def preprocess_text(self, text_queries):
            toks = []
            for t in text_queries:
                if "gpt" in self.args.text_model: t = t + " <|endoftext|>"
                tok = self.tokenizer(t, add_special_tokens=True, max_length=self.args.text_len,
                                     padding="max_length", return_tensors="pt")
                toks.append({k: tok[k].reshape(-1) for k in self.token_keys})
            return {k: torch.cat([d[k].reshape(1, -1) for d in toks]) for k in toks[0]}
    from dataset import AudioDataset
    validate_pam_import_sources(pam_root)
    class DS(AudioDataset):
        def __init__(self, files): self.filelist = files; self.repro = False
        def get_files(self): return self.filelist
        def readaudio(self, file):
            x, sr = sf.read(file, dtype="float32", always_2d=True); assert sr == 44100, (file, sr)
            return self.process_audio(torch.from_numpy(x[:, 0].copy()))
    return PAMFixed(use_cuda=False), DS

def wav_of(d):
    d = require_directory(d, "CELLO_DATA_ROOT (take directory)")
    w = sorted(glob.glob(str(d / "*.wav")))
    if len(w) != 1:
        raise FileNotFoundError(f"Expected one WAV in {d}; found {len(w)}. Check CELLO_DATA_ROOT.")
    return w[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); a = ap.parse_args()
    conds = {k: v for k, v in MANIFEST.items() if not a.only or any(s in k for s in a.only.split(","))}
    dirs = sorted({d for v in conds.values() for pr in v["pairs"] for d in pr})
    for d in dirs:
        wav_of(d)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(conds)} conditions, {len(dirs)} takes")
    scores = {}
    t = time.time(); ab = load_audiobox()
    for d in dirs:
        o = ab.forward([{"path": wav_of(d)}])[0]
        scores[d.name] = {k: float(v) for k, v in o.items()}
    print(f"audiobox done {time.time()-t:.0f}s")
    t = time.time(); pam, DS = load_pam(); ds = DS([wav_of(d) for d in dirs])
    for d, i in zip(dirs, range(len(dirs))):
        f, audios, idx = ds.collate([ds[i]])
        s, seg = pam.evaluate(audios, idx)
        scores[d.name]["PAM"] = float(s[0]); scores[d.name]["PAM_chunks"] = [float(v) for v in seg[0]]
    print(f"PAM done {time.time()-t:.0f}s")
    stamp = time.strftime("%Y%m%d_%H%M")
    (OUT / f"scores_{stamp}.json").write_text(json.dumps(
        {"scores": scores, "manifest": {k: [(b.name, r.name) for b, r in v["pairs"]] for k, v in conds.items()}}, indent=1))
    # ---------- tables ----------
    AX = ["PQ", "CE", "CU", "PC", "PAM"]
    print("\nPer-take (Audiobox PQ/CE/CU/PC 1-10; PAM 0-1)")
    for k, v in conds.items():
        print(f"\n## {k}")
        for b, r in v["pairs"]:
            sb, sr_ = scores[b.name], scores[r.name]
            print("  Base " + b.name[-6:] + "  " + "  ".join(f"{x}={sb[x]:.3f}" for x in AX))
            print("  RL   " + r.name[-6:] + "  " + "  ".join(f"{x}={sr_[x]:.3f}" for x in AX))
        print("  paired delta RL-Base:")
        for x in AX:
            ds_ = [scores[r.name][x] - scores[b.name][x] for b, r in v["pairs"]]
            sd = st.stdev(ds_) if len(ds_) > 1 else float("nan")
            pos = sum(1 for z in ds_ if z > 0)
            print(f"    {x:3s} {st.mean(ds_):+.3f} (SD {sd:.3f}) {pos}/{len(ds_)} positive  [" + ", ".join(f"{z:+.3f}" for z in ds_) + "]")

if __name__ == "__main__":
    main()
