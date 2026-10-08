"""
analyze_clap_vs_acoustics.py -- why do the CLAP prompt pairs disagree?
Per take: dBFS, spectral centroid, harmonic/percussive energy ratio (HPSS,
margin 3), commanded depth/speed from *_performance.json. (1) paired
RL-Base means per condition; (2) within-piece Spearman of each pair score
vs each feature over the live twinkle takes. the operator, 2026-09-02.
"""
import json, glob, numpy as np, soundfile as sf, librosa, score_takes as S
from scipy.stats import spearmanr
sc = json.load(open(sorted(S.OUT.glob("clap_prompts_*.json"))[-1]))["scores"]
def take_stats(d):
    x, sr = sf.read(str(S.wav_of(d)), dtype="float32"); x = x if x.ndim == 1 else x[:, 0]
    Sx = np.abs(librosa.stft(x, n_fft=2048, hop_length=512)); H, P = librosa.decompose.hpss(Sx, margin=3.0)
    f = sorted(glob.glob(str(d / "*_performance.json"))); tl = json.load(open(f[0]))["timeline"] if f else []
    return dict(rms=20*np.log10(np.sqrt(np.mean(x**2))+1e-9), cent=float(np.mean(librosa.feature.spectral_centroid(S=Sx, sr=sr))),
                hp=10*np.log10((H**2).sum()/((P**2).sum()+1e-9)), dur=len(x)/sr,
                dep=np.mean([t["depth_m"] for t in tl])*1000 if tl else np.nan, spd=np.mean([t["mean_speed"] for t in tl]) if tl else np.nan)
st = {d.name: take_stats(d) for v in S.MANIFEST.values() for pr in v["pairs"] for d in pr}
print(f"{'cond':26s} {'ΔdBFS':>7s} {'Δcent':>7s} {'ΔH/P dB':>8s} {'Δdepth mm':>10s} {'Δspeed':>8s}")
for k, v in S.MANIFEST.items():
    m = np.mean([[st[r.name][q]-st[b.name][q] for q in ["rms","cent","hp","dep","spd"]] for b, r in v["pairs"]], axis=0)
    print(f"{k[:26]:26s} {m[0]:+7.2f} {m[1]:+7.0f} {m[2]:+8.2f} {m[3]:+10.2f} {m[4]:+8.4f}")
names = [n for n in st if 30 < st[n]["dur"] < 60]
print(f"\nwithin-piece Spearman, live twinkle takes n={len(names)}")
for pair in ["N_clean","T_tone","D_dyn","P_proc"]:
    y = [sc[n][pair] for n in names]
    print(f"  {pair:8s} " + "  ".join(f"{q} {spearmanr([st[n][q] for n in names], y).correlation:+.2f}" for q in ["rms","cent","hp"]))
