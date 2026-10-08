"""Mean level and periodicity from 150 ms before to 400 ms after the logged stroke start t_b (long notes, 9 pairs pooled)."""
import sys, os, math, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acoustic_diag_v2 as A
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T0, T1 = -0.150, 0.400
def curves(tid):
    y, sr, perf = A.load_take(tid); lead = perf.get('audio_lead_in_s', 0.0); tl = perf['timeline']; L = []; P = []
    for i, n in enumerate(tl):
        if n['actual_duration'] < 0.4 or i == 0: continue
        tb = lead + n['actual_onset']; fr = A.frames(y, sr, tb + T0, tb + T1)
        if len(fr) < 100: continue
        L.append([20*math.log10(max(r, 1e-9)) for _, r, _, _ in fr][:100]); P.append([p for _, _, p, _ in fr][:100])
    return np.array(L), np.array(P)
fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
for c, (piece, logs) in enumerate(A.LOGS.items()):
    LB, PB, LR, PR = [], [], [], []
    for log in logs:
        bases, rls = A.takes_from(log)
        for t in bases: l, p = curves(t); LB.append(l); PB.append(p)
        for t in rls: l, p = curves(t); LR.append(l); PR.append(p)
    LB, PB, LR, PR = [np.concatenate(x) for x in (LB, PB, LR, PR)]
    t = (np.arange(LB.shape[1])*A.HOP + T0 + A.WIN/2)*1000
    ax = axes[0, c]; ax.plot(t, LB.mean(0), 'k-', label=f'Base (n={len(LB)})'); ax.plot(t, LR.mean(0), 'r-', label=f'RL (n={len(LR)})'); ax.axvline(0, color='gray', lw=0.8); ax.set_title(piece); ax.legend(fontsize=7); ax.grid(alpha=.3)
    if c == 0: ax.set_ylabel('level dBFS (20 ms frames)')
    ax = axes[1, c]; ax.plot(t, PB.mean(0), 'k-'); ax.plot(t, PR.mean(0), 'r-'); ax.axvline(0, color='gray', lw=0.8); ax.grid(alpha=.3); ax.set_xlabel('ms from logged stroke start'); ax.set_ylim(0.3, 1.0)
    if c == 0: ax.set_ylabel('short-time periodicity at f0')
fig.suptitle('Long notes, 9 live pairs per piece, aligned at the logged stroke start (bow change after a stop): Base black, RL red')
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'acoustic_transition_v2_20260906.png'), dpi=130); print('ok')
