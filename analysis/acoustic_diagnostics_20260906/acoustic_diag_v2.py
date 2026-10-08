"""Acoustic diagnostics v2 on the 45 live A/B pairs (RL - Base), classifier-free.
Changes vs v1 (review 9/6): (1) bow-transition analysis anchored at the logged stroke start t_b
(the robot's actual_onset; strokes alternate direction, so every stroke start is a bow change after a
stopped gap); fixed windows: gap [-150, 0] ms, W1 [0, 50], W2 [50, 150], W3 [150, 300]; lock-in time
from t_b; 'fraction above 100 ms' (no 'abnormal'). (2) Sustain HNR counts harmonics up to the band
edge (k f0 <= 8 kHz), so high harmonics are not counted as noise. (3) Results per training seed
(mean of that seed's 3 pairs) with 9-pair sign counts as a supplement; 'repeat scale' = mean
|Base(k+1) - Base(k)|, a reference for baseline repeat differences, not a significance threshold.
(4) RMS-adjusted HNR difference kept as exploratory (regression intercept), not an equal-loudness test.
CELLO_DATA_ROOT selects local logs and perform_* takes; the default is this
checkout's rl/checkpoints_piece. Historical pairing and metrics are unchanged.
"""
import json, re, sys, os, math, statistics as st, csv
import numpy as np, soundfile as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_paths import data_root, require_file, require_directory
ROOT = str(data_root())
SP = os.path.dirname(os.path.abspath(__file__))
LOGS = {'twinkle':['abc_twinkle_2035.log','ab_twinkle_alt2_s1.log','ab_twinkle_alt2_s2.log'],
        'vocalise':['ab_vocalise_alt2_s0.log','ab_vocalise_alt2_s1.log','ab_vocalise_alt2_s2.log'],
        'maysong':['ab_maysong_alt2_s0_20260902.log','ab_maysong_m0-11_alt2_s1.log','ab_maysong_m0-11_alt2_s2.log'],
        'bach':['ab_bach_sarabande_m0-7_alt2_s0.log','ab_bach_sarabande_m0-7_alt2_s1.log','ab_bach_sarabande_m0-7_alt2_s2.log'],
        'popper':['ab_yunpiece_m9-24_alt2_s0.log','ab_yunpiece_m9-24_alt2_s1.log','ab_yunpiece_m9-24_alt2_s2.log']}
F0_NOM = 220.0; F0_LO, F0_HI = 200.0, 242.0
WIN, HOP = 0.020, 0.005            # s, periodicity frames
LOCK_PER, LOCK_HOLD = 0.85, 5      # periodicity threshold, consecutive frames (25 ms)
LOCK_THR_MS = 100.0
WINDOWS = {'gap': (-0.150, 0.0), 'w1': (0.0, 0.050), 'w2': (0.050, 0.150), 'w3': (0.150, 0.300)}
SUS_HEAD, SUS_TAIL, SUS_MIN = 0.150, 0.100, 0.250
SUS_WIN, SUS_HOP = 4096, 1024; BAND = (100.0, 8000.0); HB = 3
LONG_S = 0.5

def takes_from(log):
    cur=None; d={}
    log_path = require_file(os.path.join(ROOT, log), "CELLO_DATA_ROOT (A/B logs)")
    for line in open(log_path):
        m=re.match(r'=== (\w+)',line)
        if m: cur=m.group(1)
        m2=re.search(r'perform -> .*[/\\](perform_\d+_\d+)',line)
        if m2 and cur: d.setdefault(cur,[]).append(m2.group(1))
    rl = 'ALT2_' if 'ALT2_1' in d else 'RL'
    return [d[f'Base{k}'][0] for k in (1,2,3)], [d[f'{rl}{k}'][0] for k in (1,2,3)]

def load_take(tid):
    d = require_directory(os.path.join(ROOT, tid), "CELLO_DATA_ROOT (take directory)")
    wavs = [f for f in os.listdir(d) if f.endswith('.wav')]
    if not wavs:
        raise FileNotFoundError(f"No WAV in {d}; check CELLO_DATA_ROOT.")
    wav = wavs[0]
    perf_path = require_file(d / (wav[:-4] + '_performance.json'), "CELLO_DATA_ROOT (performance metadata)")
    y, sr = sf.read(os.path.join(d, wav), dtype='float32', always_2d=True); y = y.mean(axis=1)
    perf = json.load(open(perf_path))
    return y, sr, perf


def periodicity(frame, sr):
    x = frame - frame.mean(); n = len(x)
    lo, hi = int(sr/F0_HI), int(math.ceil(sr/F0_LO)); m = n - hi
    if m < 64: return 0.0, float('nan')
    x0 = x[:m]; e0 = float(np.dot(x0, x0))
    if e0 < 1e-12: return 0.0, float('nan')
    Y = np.lib.stride_tricks.sliding_window_view(x, m)[lo:hi+1]
    r = (Y @ x0) / (np.sqrt(e0 * np.einsum('ij,ij->i', Y, Y)) + 1e-12)
    k = int(np.argmax(r)); lag = lo + k
    if 0 < k < len(r)-1:
        a,b,c = r[k-1],r[k],r[k+1]; d = a-2*b+c; lag = lag + (0.5*(a-c)/d if d != 0 else 0.0)
    return float(r[k]), float(sr/lag)

def frames(y, sr, ta, tb):
    """20 ms frames fully inside [ta, tb] -> lists of (t_center, rms, per, f0)"""
    n = int(WIN*sr); h = int(HOP*sr); out = []
    i = int(ta*sr)
    while i + n <= min(int(tb*sr), len(y)):
        fr = y[i:i+n]; p, f = periodicity(fr, sr); out.append(((i + n/2)/sr, float(np.sqrt(np.mean(fr**2))), p, f)); i += h
    return out

def transition_metrics(y, sr, t_prev_end, t_b, dur):
    out = {}
    for name, (a, b) in WINDOWS.items():
        ta, tb = t_b + a, t_b + b
        if name == 'gap':
            if t_prev_end is None: out[f'{name}_db'] = float('nan'); out[f'{name}_per'] = float('nan'); continue
            ta = max(ta, t_prev_end)
        if tb > t_b + dur + 1e-6 and name != 'gap': out[f'{name}_db'] = float('nan'); out[f'{name}_per'] = float('nan'); continue
        fr = frames(y, sr, ta, tb)
        if len(fr) < 2: out[f'{name}_db'] = float('nan'); out[f'{name}_per'] = float('nan'); continue
        out[f'{name}_db'] = 20*math.log10(max(math.sqrt(st.mean(r**2 for _, r, _, _ in fr)), 1e-9))
        out[f'{name}_per'] = st.mean(p for _, _, p, _ in fr)
    fr = frames(y, sr, t_b, t_b + min(dur, 0.6))
    if len(fr) >= LOCK_HOLD:
        lvl = 0.10*max(r for _, r, _, _ in fr)
        ok = [(p >= LOCK_PER) and (F0_LO <= f <= F0_HI) and (r >= lvl) for _, r, p, f in fr]
        j = next((j for j in range(len(ok)-LOCK_HOLD+1) if all(ok[j:j+LOCK_HOLD])), None)
        ms = (fr[j][0] - t_b)*1000.0 if j is not None else min(dur, 0.6)*1000.0
        out['lock_ms'] = ms; out['lock_gt100'] = float(ms > LOCK_THR_MS); out['lock_never'] = float(j is None)
    else:
        out['lock_ms'] = float('nan'); out['lock_gt100'] = float('nan'); out['lock_never'] = float('nan')
    return out

def sustain_metrics(y, sr, t0, dur):
    a = t0 + SUS_HEAD; b = t0 + dur - SUS_TAIL
    if b - a < SUS_MIN: return None
    seg = y[int(a*sr):int(b*sr)]; win = np.hanning(SUS_WIN); freqs = np.fft.rfftfreq(SUS_WIN, 1/sr)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1]); w = HB*sr/SUS_WIN
    hnrs = []; nhr = []; cents = []; rmss = []; f0s = []
    for i in range(0, len(seg) - SUS_WIN + 1, SUS_HOP):
        fr = seg[i:i+SUS_WIN]; p, f0 = periodicity(fr[:2048], sr)
        if not (F0_LO <= f0 <= F0_HI): f0 = F0_NOM
        P = np.abs(np.fft.rfft(fr*win))**2; tot = float(P[band].sum())
        if tot <= 0: continue
        harm = np.zeros_like(P, dtype=bool)
        for k in range(1, int(BAND[1]/f0) + 1):        # every harmonic inside the band
            fk = k*f0; harm |= (freqs >= fk - w) & (freqs <= fk + w)
        H = float(P[harm & band].sum()); N = max(tot - H, 1e-12)
        hnrs.append(10*math.log10(H/N)); nhr.append(N/tot); cents.append(float((freqs[band]*P[band]).sum()/tot))
        rmss.append(float(np.sqrt(np.mean(fr**2)))); f0s.append(f0)
    if len(hnrs) < 3: return None
    return dict(hnr_db=st.mean(hnrs), hnr_sd_db=st.pstdev(hnrs), nonharm=st.mean(nhr), centroid_hz=st.mean(cents),
                rms_dbfs=20*math.log10(max(st.mean(rmss), 1e-9)), f0_hz=st.mean(f0s), n_frames=len(hnrs))

def analyze_take(tid):
    y, sr, perf = load_take(tid); lead = perf.get('audio_lead_in_s', 0.0); notes = []; tl = perf['timeline']
    for i, n in enumerate(tl):
        t_b = lead + n['actual_onset']; dur = n['actual_duration']
        if t_b + 0.05 > len(y)/sr: continue
        prev_end = (lead + tl[i-1]['actual_onset'] + tl[i-1]['actual_duration']) if i > 0 else None
        row = dict(note_index=n['note_index'], dur_s=dur, long=dur >= LONG_S, gap_ms=((t_b - prev_end)*1000.0 if prev_end else float('nan')))
        row.update(transition_metrics(y, sr, prev_end, t_b, dur))
        s = sustain_metrics(y, sr, t_b, dur)
        if s: row.update(s)
        notes.append(row)
    return notes

KEYS_TR = ['gap_db', 'gap_per', 'w1_db', 'w1_per', 'w2_db', 'w2_per', 'w3_db', 'w3_per', 'lock_ms', 'lock_gt100']
KEYS_SUS = ['rms_dbfs', 'hnr_db', 'hnr_sd_db', 'nonharm', 'centroid_hz']

def take_summary(notes, which):
    sel = [n for n in notes if n['long'] == (which == 'long')]; out = {'n': len(sel)}
    for k in KEYS_TR + (KEYS_SUS if which == 'long' else []):
        v = [n[k] for n in sel if k in n and n[k] == n[k]]; out[k] = st.mean(v) if v else float('nan')
    return out

def diff(a, b): return {k: (a[k] - b[k]) for k in a if k != 'n'}

FMT = {'gap_db': '{:+6.2f}', 'w1_db': '{:+6.2f}', 'w2_db': '{:+6.2f}', 'w3_db': '{:+6.2f}', 'gap_per': '{:+6.3f}', 'w1_per': '{:+6.3f}', 'w2_per': '{:+6.3f}', 'w3_per': '{:+6.3f}',
       'lock_ms': '{:+6.1f}', 'lock_gt100': '{:+6.3f}', 'rms_dbfs': '{:+6.2f}', 'hnr_db': '{:+6.2f}', 'hnr_sd_db': '{:+6.2f}', 'nonharm': '{:+7.4f}', 'centroid_hz': '{:+6.0f}'}
def f(k, v): return '   nan' if v != v else FMT[k].format(v)

if __name__ == '__main__':
    cache = {}; results = []; floors = []
    for piece, logs in LOGS.items():
        for seed, log in enumerate(logs):
            bases, rls = takes_from(log)
            for tid in bases + rls:
                if tid not in cache: cache[tid] = analyze_take(tid)
            for k in range(3):
                b, r = cache[bases[k]], cache[rls[k]]; bi = {n['note_index']: n for n in b}
                pn = [(rn['hnr_db'] - bi[rn['note_index']]['hnr_db'], rn['rms_dbfs'] - bi[rn['note_index']]['rms_dbfs'])
                      for rn in r if 'hnr_db' in rn and rn['note_index'] in bi and 'hnr_db' in bi[rn['note_index']]]
                results.append(dict(piece=piece, seed=seed, pair=k+1, base=bases[k], rl=rls[k],
                                    long=diff(take_summary(r, 'long'), take_summary(b, 'long')), short=diff(take_summary(r, 'short'), take_summary(b, 'short')),
                                    base_long=take_summary(b, 'long'), base_short=take_summary(b, 'short'), note_deltas=pn))
            for k in range(2):
                floors.append(dict(piece=piece, seed=seed, long=diff(take_summary(cache[bases[k+1]], 'long'), take_summary(cache[bases[k]], 'long')),
                                   short=diff(take_summary(cache[bases[k+1]], 'short'), take_summary(cache[bases[k]], 'short'))))
    json.dump(dict(results=results, floors=floors, per_take=cache, params=dict(WIN=WIN, HOP=HOP, LOCK_PER=LOCK_PER, LOCK_HOLD=LOCK_HOLD, LOCK_THR_MS=LOCK_THR_MS,
              WINDOWS=WINDOWS, SUS_HEAD=SUS_HEAD, SUS_TAIL=SUS_TAIL, SUS_MIN=SUS_MIN, BAND=BAND, HB=HB, LONG_S=LONG_S)), open(os.path.join(SP, 'acoustic_diag_v2_45pairs.json'), 'w'))
    with open(os.path.join(SP, 'acoustic_diag_v2_45pairs.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['piece', 'seed', 'pair', 'base', 'rl'] + ['dL_'+k for k in KEYS_TR+KEYS_SUS] + ['dS_'+k for k in KEYS_TR])
        for x in results: w.writerow([x['piece'], x['seed'], x['pair'], x['base'], x['rl']] + [x['long'][k] for k in KEYS_TR+KEYS_SUS] + [x['short'][k] for k in KEYS_TR])
    for which, keys in (('long', KEYS_TR + KEYS_SUS), ('short', KEYS_TR)):
        print(f"\n===== {which.upper()} notes, RL - Base: per training seed (mean of its 3 pairs) s0 | s1 | s2 ; [pairs>0/9] ; repeat scale = mean |Base(k+1)-Base(k)| (6 per piece)")
        for k in keys:
            print(f"-- {k}")
            for piece in LOGS:
                rs = [x for x in results if x['piece'] == piece and x[which][k] == x[which][k]]
                fl = [abs(x[which][k]) for x in floors if x['piece'] == piece and x[which][k] == x[which][k]]
                if not rs: print(f"   {piece:9s} (none)"); continue
                seeds = [st.mean(x[which][k] for x in rs if x['seed'] == s) if any(x['seed'] == s for x in rs) else float('nan') for s in range(3)]
                pos = sum(1 for x in rs if x[which][k] > 0)
                print(f"   {piece:9s} {' | '.join(f(k, v) for v in seeds)}   [{pos}/{len(rs)}]   scale {f(k, st.mean(fl)).strip() if fl else 'n/a'}   base {f(k, st.mean(x['base_'+which][k] for x in rs if x['base_'+which][k] == x['base_'+which][k])).strip()}   n/take {rs[0]['base_'+which]['n']}")
    print("\n===== exploratory: RMS-adjusted HNR difference = intercept of dHNR = a + b dRMS over matched long notes (pooled per piece); NOT an equal-loudness test")
    from scipy.stats import spearmanr
    for piece in LOGS:
        pn = [d for x in results if x['piece'] == piece for d in x['note_deltas']]
        if len(pn) < 5: continue
        h = np.array([d[0] for d in pn]); r = np.array([d[1] for d in pn]); b, a = np.polyfit(r, h, 1); rho, p = spearmanr(h, r)
        print(f"   {piece:9s} n={len(pn):3d} rho {rho:+.2f} slope {b:+.2f} dB/dB intercept {a:+.2f} dB | mean dHNR {h.mean():+.2f} mean dRMS {r.mean():+.2f}")
    print("\n===== baseline lock-in distribution from t_b (all notes, 9 Base takes): p10/p50/p90 ms, frac>100, never")
    for piece in LOGS:
        v = [n['lock_ms'] for x in results for n in cache[x['base']] if x['piece'] == piece and n.get('lock_ms') == n.get('lock_ms')]
        nv = [n['lock_never'] for x in results for n in cache[x['base']] if x['piece'] == piece and n.get('lock_never') == n.get('lock_never')]
        q = np.percentile(v, [10, 50, 90]); print(f"   {piece:9s} n={len(v):4d} {q[0]:6.1f}/{q[1]:6.1f}/{q[2]:6.1f}  >100 {np.mean(np.array(v) > 100):.2f}  never {st.mean(nv):.3f}  gap ms median {st.median(n['gap_ms'] for x in results for n in cache[x['base']] if x['piece'] == piece and n['gap_ms'] == n['gap_ms']):.0f}")
