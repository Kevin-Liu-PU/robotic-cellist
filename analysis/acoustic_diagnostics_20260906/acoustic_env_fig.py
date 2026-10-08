"""Mean level and periodicity curves after the acoustic onset, Base vs RL, per piece (long notes, 9 pairs pooled)."""
import sys, os, math, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acoustic_diag as A
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T = 0.50; hop = A.ATT_HOP; nfr = int(T/hop)
def curves(tid):
    y, sr, perf = A.load_take(tid); lead = perf.get('audio_lead_in_s', 0.0); n_win = int(A.ATT_WIN*sr); h = int(hop*sr)
    L = []; P = []
    for n in perf['timeline']:
        if n['actual_duration'] < A.LONG_S: continue
        t0 = lead + n['actual_onset']; i = int(t0*sr); rms = []; per = []
        while i + n_win <= min(int((t0+0.7)*sr), len(y)):
            fr = y[i:i+n_win]; p, f = A.periodicity(fr, sr); rms.append(float(np.sqrt(np.mean(fr**2)))); per.append(p); i += h
        if len(rms) < nfr + 5: continue
        mx = max(rms); on = next((j for j, r in enumerate(rms) if r >= 0.1*mx), 0)
        if on + nfr > len(rms): continue
        L.append(20*np.log10(np.maximum(rms[on:on+nfr], 1e-9))); P.append(per[on:on+nfr])
    return np.array(L), np.array(P)
fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
out = {}
for c, (piece, logs) in enumerate(A.LOGS.items()):
    LB, PB, LR, PR = [], [], [], []
    for log in logs:
        bases, rls = A.takes_from(log)
        for t in bases:
            l, p = curves(t); LB.append(l); PB.append(p)
        for t in rls:
            l, p = curves(t); LR.append(l); PR.append(p)
    LB, PB, LR, PR = [np.concatenate(x) for x in (LB, PB, LR, PR)]
    t = np.arange(nfr)*hop*1000
    ax = axes[0, c]; ax.plot(t, LB.mean(0), 'k-', label=f'Base (n={len(LB)})'); ax.plot(t, LR.mean(0), 'r-', label=f'RL (n={len(LR)})'); ax.set_title(piece); ax.legend(fontsize=7); ax.grid(alpha=.3)
    if c == 0: ax.set_ylabel('level dBFS (20 ms frames)')
    ax = axes[1, c]; ax.plot(t, PB.mean(0), 'k-'); ax.plot(t, PR.mean(0), 'r-'); ax.grid(alpha=.3); ax.set_xlabel('ms after acoustic onset'); ax.set_ylim(0.3, 1.0)
    if c == 0: ax.set_ylabel('periodicity at f0 (norm. corr.)')
    out[piece] = dict(t=t.tolist(), LB=LB.mean(0).tolist(), LR=LR.mean(0).tolist(), PB=PB.mean(0).tolist(), PR=PR.mean(0).tolist())
fig.suptitle('Long notes, 9 live pairs per piece: mean level and periodicity after the acoustic onset (Base black, RL red)')
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'acoustic_envelopes_20260906.png'), dpi=130)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'acoustic_envelopes.json'), 'w'))
for piece, d in out.items():
    LB, LR, PB, PR = (np.array(d[k]) for k in ('LB', 'LR', 'PB', 'PR'))
    print(f"{piece:9s} level RL-Base @ 25/50/100/200/400 ms: " + ' '.join(f'{LR[i]-LB[i]:+.1f}' for i in (5, 10, 20, 40, 80)) + " dB | periodicity RL-Base: " + ' '.join(f'{PR[i]-PB[i]:+.3f}' for i in (5, 10, 20, 40, 80)))
