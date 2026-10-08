"""Score take directories with the FROZEN audio-only tone model C, per executed note, and report paired deltas.
    .venv/bin/python score_takes.py --model model_C.joblib --pairs45            # the 45 live A/B pairs of Table II
    .venv/bin/python score_takes.py --model model_C.joblib --d-tsv rl/checkpoints_piece/abD_vocalise_takes_20260908.tsv
    .venv/bin/python score_takes.py --model model_C.joblib --dirs DIR_A DIR_B   # any take dirs
Writes posthoc_C.json into each take dir (per note: tone_C, dur_s). Deltas: RL - Base per pair (note-matched means),
per training seed, per piece, [pairs > 0]; long (>= 0.5 s) and short notes separately. No classifier is used.
Revision: 2026-09-06."""
import argparse, json, os, re, statistics as st, sys
import numpy as np, joblib
import features_c as F
CP = F.DATA_ROOT
LOGS = {'twinkle':['abc_twinkle_2035.log','ab_twinkle_alt2_s1.log','ab_twinkle_alt2_s2.log'],
        'vocalise':['ab_vocalise_alt2_s0.log','ab_vocalise_alt2_s1.log','ab_vocalise_alt2_s2.log'],
        'maysong':['ab_maysong_alt2_s0_20260902.log','ab_maysong_m0-11_alt2_s1.log','ab_maysong_m0-11_alt2_s2.log'],
        'bach':['ab_bach_sarabande_m0-7_alt2_s0.log','ab_bach_sarabande_m0-7_alt2_s1.log','ab_bach_sarabande_m0-7_alt2_s2.log'],
        'popper':['ab_yunpiece_m9-24_alt2_s0.log','ab_yunpiece_m9-24_alt2_s1.log','ab_yunpiece_m9-24_alt2_s2.log']}

def takes_from(log):
    cur = None; d = {}
    log_path = F.require_file(os.path.join(CP, log), "CELLO_DATA_ROOT (A/B logs)")
    for line in open(log_path):
        m = re.match(r'=== (\w+)', line)
        if m: cur = m.group(1)
        m2 = re.search(r'perform -> .*[/\\](perform_\d+_\d+)', line)
        if m2 and cur: d.setdefault(cur, []).append(m2.group(1))
    rl = 'ALT2_' if 'ALT2_1' in d else 'RL'
    return [d[f'Base{k}'][0] for k in (1, 2, 3)], [d[f'{rl}{k}'][0] for k in (1, 2, 3)]

def score_dir(d, model, meta, force=False):
    d = d if os.path.isabs(d) else os.path.join(CP, d); out = os.path.join(d, 'posthoc_C.json')
    if os.path.exists(out) and not force:
        j = json.load(open(out))
        if j.get('feature_version') == meta['feature_version'] and j.get('model_frozen_at') == meta['frozen_at']: return j
    notes = []
    for n in F.take_note_segments(d):
        v = np.nan_to_num(F.segment_features(n['audio'], n['sr']))[None, :]
        notes.append(dict(note_index=n['note_index'], dur_s=round(n['dur_s'], 4), tone_C=round(float(model.predict(v)[0]), 4)))
    j = dict(dir=os.path.basename(d), n=len(notes), mean_tone_C=st.mean(x['tone_C'] for x in notes) if notes else None,
             feature_version=meta['feature_version'], model_frozen_at=meta['frozen_at'], model=meta['model'], per_note=notes)
    json.dump(j, open(out, 'w')); return j

def paired(jr, jb):
    b = {x['note_index']: x for x in jb['per_note']}; out = {}
    for which, sel in (('all', lambda x: True), ('long', lambda x: x['dur_s'] >= 0.5), ('short', lambda x: x['dur_s'] < 0.5)):
        ds = [x['tone_C'] - b[x['note_index']]['tone_C'] for x in jr['per_note'] if x['note_index'] in b and sel(x)]
        out[which] = st.mean(ds) if ds else float('nan'); out['n_' + which] = len(ds)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--pairs45', action='store_true')
    ap.add_argument('--d-tsv', default=None); ap.add_argument('--dirs', nargs='*', default=[]); ap.add_argument('--force', action='store_true')
    a = ap.parse_args(); M = joblib.load(a.model); model, meta = M['model'], M['meta']
    print(f"model C: {meta['model']} target {meta['target']} n={meta['n']} oof Spearman {meta['report'][meta['model']]['spearman']:+.3f} frozen {meta['frozen_at']}")
    for d in a.dirs:
        j = score_dir(d, model, meta, a.force); print(f"{j['dir']} n={j['n']} mean tone_C {j['mean_tone_C']:.4f}")
    if a.pairs45:
        rows = []
        for piece, logs in LOGS.items():
            for seed, log in enumerate(logs):
                bases, rls = takes_from(log)
                for k in range(3):
                    jb = score_dir(bases[k], model, meta, a.force); jr = score_dir(rls[k], model, meta, a.force)
                    rows.append(dict(piece=piece, seed=seed, pair=k + 1, base=bases[k], rl=rls[k], **paired(jr, jb), base_mean=jb['mean_tone_C']))
        json.dump(rows, open('deltas_C_45pairs.json', 'w'), indent=1)
        print('\n=== tone_C, RL - Base: per seed (mean of 3 pairs) s0 | s1 | s2 ; [pairs > 0 / 9] ; baseline level')
        for which in ('all', 'long', 'short'):
            print(f'-- {which} notes')
            for piece in LOGS:
                rs = [r for r in rows if r['piece'] == piece and r[which] == r[which]]
                if not rs: continue
                seeds = [st.mean(r[which] for r in rs if r['seed'] == s) for s in range(3) if any(r['seed'] == s for r in rs)]
                print(f"   {piece:9s} {' | '.join(f'{v:+.4f}' for v in seeds)}   [{sum(1 for r in rs if r[which] > 0)}/{len(rs)}]   base {st.mean(r['base_mean'] for r in rs):.3f}")
    if a.d_tsv:
        print('\n=== ablation D (vocalise): tone_C deltas per seed, mean of 3 rounds [rounds > 0]')
        for line in open(a.d_tsv):
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            seed = parts[0].split()[1]; arms = {p.split()[0]: p.split()[1:] for p in parts[1:]}
            J = {arm: [score_dir(d, model, meta, a.force) for d in dirs] for arm, dirs in arms.items()}
            cmps = [('FULL', 'Base'), ('ABL', 'Base'), ('FULL', 'ABL')]
            if 'FULLNEW' in J: cmps += [('FULLNEW', 'Base'), ('FULLNEW', 'ABL'), ('FULLNEW', 'FULL')]   # same-day FULL control (seed 0)
            for x, y in cmps:
                if x not in J or y not in J: continue
                ds = [paired(J[x][i], J[y][i])['all'] for i in range(3)]
                print(f"   seed {seed} {x + '-' + y:13s} {st.mean(ds):+.4f}  [{sum(1 for v in ds if v > 0)}/3]  rounds {' '.join(f'{v:+.4f}' for v in ds)}")
if __name__ == '__main__': main()
