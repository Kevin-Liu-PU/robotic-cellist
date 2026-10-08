"""
score_clap_prompts.py -- CLAP antonym prompting with fixed pairs.
Same model/chunking as PAM; only the prompt pair
changes. Reuses score_takes.py's manifest and loaders.

  /path/to/judge-venv/bin/python score_clap_prompts.py

"""
import sys, json, time, statistics as st
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_takes as S

PAIRS = {
  "N_clean":  ["the sound is clear and clean.", "the sound is noisy and with artifacts."],  # PAM original
  "T_tone":   ["a cello played with a warm, clean, resonant bowed tone",
               "a cello played with a scratchy, harsh, noisy bow tone"],
  "D_dyn":    ["a cello performance with clear dynamic contrast between soft and loud passages",
               "a cello performance played at one flat, unchanging volume"],
  # second registration, daylog 9/2 07:1x (after T/D results; from the action space)
  "P_proc":   ["a cello played with firm, steady bow contact: every note speaks fully and is sustained evenly at a controlled volume from start to end",
               "a cello played with unsteady bow contact: notes that are weak and airy, stall or fade mid-note, or are pressed into a scratchy, choked sound"],
}

def main():
    conds = S.MANIFEST
    dirs = sorted({d for v in conds.values() for pr in v["pairs"] for d in pr})
    pam, DS = S.load_pam()
    ds = DS([S.wav_of(d) for d in dirs])
    audio = [ds.collate([ds[i]]) for i in range(len(dirs))]   # (files, tensor, idx)
    scores = {d.name: {} for d in dirs}
    for name, prompts in PAIRS.items():
        pam.pam_prompts = prompts; pam.get_text_embeddings()
        for d, (f, a, idx) in zip(dirs, audio):
            s, seg = pam.evaluate(a, idx)
            scores[d.name][name] = float(s[0])
    stamp = time.strftime("%Y%m%d_%H%M")
    (S.OUT / f"clap_prompts_{stamp}.json").write_text(json.dumps({"pairs": PAIRS, "scores": scores}, indent=1))
    print("pairs:"); [print(f"  {k}: + '{v[0]}'  /  - '{v[1]}'") for k, v in PAIRS.items()]
    print("\nPaired delta RL-Base (mean, SD, positives/pairs); Base level in brackets")
    for k, v in conds.items():
        print(f"\n## {k}")
        for name in PAIRS:
            ds_ = [scores[r.name][name] - scores[b.name][name] for b, r in v["pairs"]]
            base = st.mean(scores[b.name][name] for b, r in v["pairs"])
            sd = st.stdev(ds_) if len(ds_) > 1 else float("nan")
            print(f"  {name:8s} {st.mean(ds_):+.3f} (SD {sd:.3f}) {sum(z>0 for z in ds_)}/{len(ds_)}  [Base {base:.3f}]  " + ", ".join(f"{z:+.3f}" for z in ds_))

if __name__ == "__main__":
    main()
