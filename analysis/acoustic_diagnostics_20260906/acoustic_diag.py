"""Acoustic diagnostics on the 45 live A/B pairs (RL - Base), classifier-free.
Per note: attack settling time (onset -> stable periodic vibration at the string's f0),
abnormal-attack flag; for notes with >=0.5 s: sustain HNR (harmonic-to-noise ratio, dB),
its within-note fluctuation (sd, dB), non-harmonic energy ratio, sustain RMS (dBFS),
spectral centroid. Attack/release segments excluded from sustain (first 150 ms, last 100 ms).
Not combined into a score. Baseline-vs-baseline differences give the noise floor.
CELLO_DATA_ROOT selects local logs and perform_* takes; the default is this
checkout's rl/checkpoints_piece. Historical pairing and metrics are unchanged.
"""
import json, re, sys, os, math, statistics as st
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
ATT_WIN, ATT_HOP = 0.020, 0.005      # s: periodicity tracking in the attack
SETTLE_PER, SETTLE_HOLD = 0.85, 5     # periodicity threshold, consecutive frames (25 ms)
ABNORMAL_MS = 100.0                   # settling slower than this = abnormal attack
SUS_HEAD, SUS_TAIL, SUS_MIN = 0.150, 0.100, 0.250   # s
SUS_WIN, SUS_HOP = 4096, 1024         # samples (93 ms / 23 ms at 44.1k)
NH = 25                               # harmonics counted (up to 5.5 kHz)
HB = 3                                # harmonic band half-width in bins (~32 Hz)
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
    """max normalized cross-correlation between x[0:m] and x[l:l+m] over lags l in the f0 range
    (m = n - lmax so every lag uses the same overlap) -> (per, f0)"""
    x = frame - frame.mean(); n = len(x)
    lo, hi = int(sr/F0_HI), int(math.ceil(sr/F0_LO))
    m = n - hi
    if m < 64: return 0.0, float('nan')
    x0 = x[:m]; e0 = float(np.dot(x0, x0))
    if e0 < 1e-12: return 0.0, float('nan')
    Y = np.lib.stride_tricks.sliding_window_view(x, m)[lo:hi+1]     # rows = lags lo..hi
    num = Y @ x0; den = np.sqrt(e0 * np.einsum('ij,ij->i', Y, Y)) + 1e-12
    r = num / den; k = int(np.argmax(r)); lag = lo + k
    if 0 < k < len(r)-1:
        a,b,c = r[k-1],r[k],r[k+1]; d = a-2*b+c
        lag = lag + (0.5*(a-c)/d if d != 0 else 0.0)
    return float(r[k]), float(sr/lag)

def attack_metrics(y, sr, t0, dur):
    """t0 = robot motion onset. Acoustic onset = first 20 ms frame with RMS >= 10% of the note's max
    (within the first 0.6 s). Settling = acoustic onset -> first 25 ms run of frames that are periodic
    at the string's f0 (normalized correlation >= SETTLE_PER, f0 within 200-242 Hz). att_rough = mean
    (1 - periodicity) over the 150 ms after the acoustic onset. latency_ms = motion onset -> acoustic onset."""
    n_win = int(ATT_WIN*sr); hop = int(ATT_HOP*sr)
    end = min(t0 + min(dur, 0.6), len(y)/sr - ATT_WIN)
    s0 = int(t0*sr); pers = []; f0s = []; rms = []
    i = s0
    while i + n_win <= int(end*sr):
        fr = y[i:i+n_win]; p, f = periodicity(fr, sr); pers.append(p); f0s.append(f); rms.append(float(np.sqrt(np.mean(fr**2))))
        i += hop
    if len(pers) < SETTLE_HOLD + 2: return dict(settle_ms=float('nan'), abnormal=float('nan'), abnormal50=float('nan'), att_rough=float('nan'), latency_ms=float('nan'))
    lvl = max(rms) * 0.10
    on = next((j for j, r in enumerate(rms) if r >= lvl), 0)
    ok = [(p >= SETTLE_PER) and (F0_LO <= f <= F0_HI) and (r >= lvl) for p, f, r in zip(pers, f0s, rms)]
    settle = None
    for j in range(on, len(ok) - SETTLE_HOLD + 1):
        if all(ok[j:j+SETTLE_HOLD]): settle = j; break
    nr = int(0.150/ATT_HOP)
    rough = float(np.mean([1.0 - p for p in pers[on:on+nr]])) if len(pers) > on else float('nan')
    lat = on*ATT_HOP*1000.0
    mx = max(rms); r10 = next((j for j, r in enumerate(rms) if r >= 0.1*mx), 0); r90 = next((j for j, r in enumerate(rms) if r >= 0.9*mx), r10)
    rise = (r90 - r10)*ATT_HOP*1000.0
    n100 = int(0.100/ATT_HOP); att_rms = 20*math.log10(max(float(np.sqrt(np.mean(np.array(rms[on:on+n100])**2))), 1e-9))
    rough25 = float(np.mean([1.0 - p for p, r in zip(pers[on:on+nr], rms[on:on+nr]) if r >= 0.25*mx] or [float('nan')]))
    extra = dict(rise_ms=rise, att_rms_dbfs=att_rms, att_rough25=rough25)
    if settle is None:
        ms = (len(ok) - on)*ATT_HOP*1000.0
        return dict(settle_ms=ms, abnormal=1.0, abnormal50=1.0, settled=0.0, att_rough=rough, latency_ms=lat, **extra)
    ms = (settle - on)*ATT_HOP*1000.0
    return dict(settle_ms=ms, abnormal=float(ms > ABNORMAL_MS), abnormal50=float(ms > 50.0), settled=1.0, att_rough=rough, latency_ms=lat, **extra)

def sustain_metrics(y, sr, t0, dur):
    a = t0 + SUS_HEAD; b = t0 + dur - SUS_TAIL
    if b - a < SUS_MIN: return None
    seg = y[int(a*sr):int(b*sr)]
    win = np.hanning(SUS_WIN); freqs = np.fft.rfftfreq(SUS_WIN, 1/sr)
    band = (freqs >= 100) & (freqs <= 8000)
    hnrs = []; nhr = []; cents = []; rmss = []; f0s = []
    for i in range(0, len(seg) - SUS_WIN + 1, SUS_HOP):
        fr = seg[i:i+SUS_WIN]
        p, f0 = periodicity(fr[:2048], sr)
        if not (F0_LO <= f0 <= F0_HI): f0 = F0_NOM
        P = np.abs(np.fft.rfft(fr*win))**2
        tot = float(P[band].sum())
        if tot <= 0: continue
        harm = np.zeros_like(P, dtype=bool)
        for k in range(1, NH+1):
            fk = k*f0
            if fk > 8000: break
            w = HB * sr / SUS_WIN                 # fixed +-HB bins around each harmonic
            harm |= (freqs >= fk - w) & (freqs <= fk + w)
        H = float(P[harm & band].sum()); N = max(tot - H, 1e-12)
        hnrs.append(10*math.log10(H/N)); nhr.append(N/tot)
        cents.append(float((freqs[band]*P[band]).sum()/tot))
        rmss.append(float(np.sqrt(np.mean(fr**2)))); f0s.append(f0)
    if len(hnrs) < 3: return None
    return dict(hnr_db=st.mean(hnrs), hnr_sd_db=st.pstdev(hnrs), nonharm=st.mean(nhr),
                centroid_hz=st.mean(cents), rms_dbfs=20*math.log10(max(st.mean(rmss), 1e-9)), f0_hz=st.mean(f0s), n_frames=len(hnrs))

def analyze_take(tid):
    y, sr, perf = load_take(tid); lead = perf.get('audio_lead_in_s', 0.0)
    notes = []
    for n in perf['timeline']:
        t0 = lead + n['actual_onset']; dur = n['actual_duration']
        if t0 + 0.05 > len(y)/sr: continue
        row = dict(note_index=n['note_index'], dur_s=dur, long=dur >= LONG_S)
        row.update(attack_metrics(y, sr, t0, dur))
        s = sustain_metrics(y, sr, t0, dur)
        if s: row.update(s)
        notes.append(row)
    return notes

KEYS_ATT = ['settle_ms', 'abnormal', 'abnormal50', 'att_rough', 'att_rough25', 'rise_ms', 'att_rms_dbfs', 'latency_ms']
KEYS_SUS = ['hnr_db', 'hnr_sd_db', 'nonharm', 'rms_dbfs', 'centroid_hz']

def take_summary(notes, which):
    sel = [n for n in notes if n['long'] == (which == 'long')]
    out = {'n': len(sel)}
    for k in KEYS_ATT:
        v = [n[k] for n in sel if k in n and n[k] == n[k]]; out[k] = st.mean(v) if v else float('nan')
    if which == 'long':
        for k in KEYS_SUS:
            v = [n[k] for n in sel if k in n]; out[k] = st.mean(v) if v else float('nan')
        out['n_sus'] = sum(1 for n in sel if 'hnr_db' in n)
    return out

def diff(a, b):
    return {k: (a[k] - b[k]) if isinstance(a.get(k), float) and isinstance(b.get(k), float) else None for k in a if k not in ('n', 'n_sus')}

def trace(tid, note_idx, label):
    y, sr, perf = load_take(tid); lead = perf.get('audio_lead_in_s', 0.0); n = perf['timeline'][note_idx]
    t0 = lead + n['actual_onset']; n_win = int(ATT_WIN*sr); hop = int(ATT_HOP*sr); i = int(t0*sr); rows = []
    while i + n_win <= int((t0+0.35)*sr):
        fr = y[i:i+n_win]; p, f = periodicity(fr, sr); rows.append((round((i-int(t0*sr))/sr*1000), round(20*math.log10(max(float(np.sqrt(np.mean(fr**2))),1e-9)),1), round(p,2), round(f,1))); i += hop
    print(label, tid, 'note', note_idx, 'dur', round(n['actual_duration'],2), '| (ms, dBFS, per, f0) every 15 ms:')
    print('  ' + ' '.join(f'{t}:{d}/{p}/{f:.0f}' for t, d, p, f in rows[::3]))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'trace':
        trace(sys.argv[2], int(sys.argv[3]), sys.argv[4]); sys.exit()
    cache = {}
    results = []   # per pair
    floors = []    # base-vs-base
    for piece, logs in LOGS.items():
        for seed, log in enumerate(logs):
            bases, rls = takes_from(log)
            for tid in bases + rls:
                if tid not in cache: cache[tid] = analyze_take(tid)
            for k in range(3):
                b, r = cache[bases[k]], cache[rls[k]]
                # per-note matched deltas for the loudness-confound check
                bi = {n['note_index']: n for n in b}
                pn = [(rn['hnr_db'] - bi[rn['note_index']]['hnr_db'], rn['rms_dbfs'] - bi[rn['note_index']]['rms_dbfs'])
                      for rn in r if 'hnr_db' in rn and rn['note_index'] in bi and 'hnr_db' in bi[rn['note_index']]]
                results.append(dict(piece=piece, seed=seed, pair=k+1, base=bases[k], rl=rls[k],
                                    long=diff(take_summary(r, 'long'), take_summary(b, 'long')),
                                    short=diff(take_summary(r, 'short'), take_summary(b, 'short')),
                                    base_long=take_summary(b, 'long'), rl_long=take_summary(r, 'long'),
                                    base_short=take_summary(b, 'short'), rl_short=take_summary(r, 'short'),
                                    note_deltas=pn))
            for k in range(2):
                floors.append(dict(piece=piece, seed=seed, long=diff(take_summary(cache[bases[k+1]], 'long'), take_summary(cache[bases[k]], 'long')),
                                   short=diff(take_summary(cache[bases[k+1]], 'short'), take_summary(cache[bases[k]], 'short'))))
    json.dump(dict(results=results, floors=floors, per_take={t: v for t, v in cache.items()}, params=dict(
        SETTLE_PER=SETTLE_PER, SETTLE_HOLD=SETTLE_HOLD, ABNORMAL_MS=ABNORMAL_MS, SUS_HEAD=SUS_HEAD, SUS_TAIL=SUS_TAIL, SUS_MIN=SUS_MIN, LONG_S=LONG_S, NH=NH)),
        open(os.path.join(SP, 'acoustic_diag_45pairs.json'), 'w'))

    def fmt(k, v):
        if v is None or v != v: return '   n/a'
        return {'settle_ms': f'{v:+6.1f}', 'abnormal': f'{v:+6.3f}', 'abnormal50': f'{v:+6.3f}', 'att_rough': f'{v:+6.3f}', 'att_rough25': f'{v:+6.3f}', 'rise_ms': f'{v:+6.1f}', 'att_rms_dbfs': f'{v:+6.2f}', 'latency_ms': f'{v:+6.1f}', 'hnr_db': f'{v:+6.2f}', 'hnr_sd_db': f'{v:+6.2f}',
                'nonharm': f'{v:+6.4f}', 'rms_dbfs': f'{v:+6.2f}', 'centroid_hz': f'{v:+6.0f}'}[k]
    for which, keys in (('long', KEYS_ATT + KEYS_SUS), ('short', KEYS_ATT)):
        print(f"\n===== {which.upper()} notes: RL - Base, per piece = mean over 9 pairs [pairs>0 / 9] ; floor = |Base(k+1)-Base(k)| mean over 6")
        print(f"{'piece':9s} " + ' '.join(f'{k:>22s}' for k in keys))
        for piece in LOGS:
            rs = [x for x in results if x['piece'] == piece and x[which][keys[0]] is not None]
            fl = [x for x in floors if x['piece'] == piece and x[which][keys[0]] is not None]
            if not rs: print(f'{piece:9s} (no {which} notes)'); continue
            cells = []
            for k in keys:
                v = [x[which][k] for x in rs if x[which][k] is not None]; f = [abs(x[which][k]) for x in fl if x[which][k] is not None]
                pos = sum(1 for z in v if z > 0)
                cells.append(f'{fmt(k, st.mean(v))} [{pos}/{len(v)}] f{fmt(k, st.mean(f)).strip() if f else "n/a"}')
            n = rs[0]['base_' + which]['n']
            print(f'{piece:9s} ' + ' '.join(f'{c:>22s}' for c in cells) + f'   (n notes/take {n})')
    print("\n===== baseline levels (long notes, mean over the 9 Base takes): settle ms | abnormal frac | HNR dB | HNR sd | nonharm | RMS dBFS | centroid Hz | f0")
    for piece in LOGS:
        rs = [x['base_long'] for x in results if x['piece'] == piece and x['base_long']['n'] > 0]
        if not rs: continue
        f0 = st.mean(n['f0_hz'] for t in [x['base'] for x in results if x['piece'] == piece] for n in cache[t] if 'f0_hz' in n)
        print(f"{piece:9s} " + ' | '.join(f"{st.mean(x[k] for x in rs):8.3f}" for k in KEYS_ATT + KEYS_SUS) + f" | f0 {f0:.1f}")
    print("\n===== loudness confound: per-note dHNR vs dRMS (Spearman over matched long notes, pooled per piece)")
    from scipy.stats import spearmanr
    for piece in LOGS:
        pn = [d for x in results if x['piece'] == piece for d in x['note_deltas']]
        if len(pn) < 5: continue
        h, r = zip(*pn); rho, p = spearmanr(h, r)
        print(f"{piece:9s} n={len(pn):3d} rho={rho:+.2f} p={p:.3f}  mean dHNR {st.mean(h):+.2f} dB, mean dRMS {st.mean(r):+.2f} dB")
    print("\n===== per seed (long notes): dSettle ms / dAbnormal / dHNR dB / dHNRsd / dRMS dB, mean of 3 pairs")
    for piece in LOGS:
        for seed in range(3):
            rs = [x for x in results if x['piece'] == piece and x['seed'] == seed and x['long']['hnr_db'] is not None]
            if not rs: continue
            print(f"{piece:9s} s{seed} " + ' '.join(f"{k}={st.mean(x['long'][k] for x in rs):+.3f}" for k in ('settle_ms', 'abnormal', 'hnr_db', 'hnr_sd_db', 'rms_dbfs')))

    print("\n===== loudness-adjusted sustain HNR: OLS dHNR = a + b*dRMS over matched long notes per piece; a = dHNR at equal loudness")
    for piece in LOGS:
        pn = [d for x in results if x['piece'] == piece for d in x['note_deltas']]
        if len(pn) < 5: continue
        h = np.array([d[0] for d in pn]); r = np.array([d[1] for d in pn])
        b, a = np.polyfit(r, h, 1)
        # also: per-pair intercepts -> sign count
        ints = []
        for x in [x for x in results if x['piece'] == piece]:
            if len(x['note_deltas']) < 4: continue
            hh = np.array([d[0] for d in x['note_deltas']]); rr = np.array([d[1] for d in x['note_deltas']])
            bb, aa = np.polyfit(rr, hh, 1) if np.std(rr) > 1e-6 else (0.0, float(np.mean(hh)))
            ints.append(aa)
        print(f"{piece:9s} slope {b:+.2f} dB/dB  intercept {a:+.2f} dB  (per-pair intercepts mean {st.mean(ints):+.2f}, >0 in {sum(1 for v in ints if v>0)}/{len(ints)})")
    print("\n===== baseline attack settling distribution (long+short notes, 9 Base takes per piece): p10/p50/p90 ms, frac>50, frac>100, never-settled")
    for piece in LOGS:
        v = [n['settle_ms'] for x in results for n in cache[x['base']] if x['piece'] == piece and n.get('settle_ms') == n.get('settle_ms')]
        ns = [1-n.get('settled', 1.0) for x in results for n in cache[x['base']] if x['piece'] == piece and 'settle_ms' in n]
        if not v: continue
        q = np.percentile(v, [10, 50, 90])
        print(f"{piece:9s} n={len(v):4d} {q[0]:6.1f}/{q[1]:6.1f}/{q[2]:6.1f}  >50 {np.mean(np.array(v)>50):.2f}  >100 {np.mean(np.array(v)>100):.2f}  never {st.mean(ns):.3f}")
    import csv
    with open(os.path.join(SP, 'acoustic_diag_45pairs.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['piece','seed','pair','base','rl'] + ['dL_'+k for k in KEYS_ATT+KEYS_SUS] + ['dS_'+k for k in KEYS_ATT])
        for x in results:
            w.writerow([x['piece'],x['seed'],x['pair'],x['base'],x['rl']] + [x['long'][k] for k in KEYS_ATT+KEYS_SUS] + [x['short'][k] for k in KEYS_ATT])
