"""B: does each classifier's TONE head track the human tone ratings? And how well do annotators agree?
    .venv/bin/python check_b.py --manifest manifest.csv --meta DIR/metadata_human_current.jsonl [--judges both]
For every labelled recording: windows across the active region exactly as the classifier was trained
(0.5 s, hop 0.25 s, peak-normalised, window_pos = window centre within the active region, the record's 6-dim
physical vector), score_detailed() with each checkpoint, recording score = mean over windows. Reports Spearman
and MAE (in 1-4 rating levels) for tone_quality and overall, on (a) all labelled recordings and (b) the
classifier's own validation split reproduced with dataset.group_train_val_split (seed 0, 20 % of group ids),
which is the honest number for a judge that saw the rest in training. Caveat printed: the A1_A5 checkpoint's
own split/snapshot may differ; treat (b) as approximate unless its metrics json confirms the split.
Revision: 2026-09-06."""
import argparse, csv, os, sys, json
import numpy as np
from scipy.stats import spearmanr
import features_c as F
sys.path.insert(0, F.REPO); sys.path.insert(0, os.path.join(F.REPO, 'SoundClassifier'))
from SoundClassifier import dataset as ds, audio_features as af
CKPT = os.path.join(F.REPO, 'SoundClassifier', 'checkpoints')
JUDGES = {'multilength': os.path.join(CKPT, 'quality_cnn_multilength.pt'), 'human_A1_A5': os.path.join(CKPT, 'quality_cnn_human_A1_A5.pt')}

def recording_scores(judge, path, record):
    y, sr = F.load_audio(path); a, b = ds._active_region(record, len(y), sr); a_n, b_n = int(a * sr), int(b * sr)
    phys = ds.build_physical_features(record); heads = []
    for chunk, s, e in ds.cut_windows(y, sr, record):
        pos = float(((s + e) / 2 - a_n) / max(b_n - a_n, 1)); chunk = af.normalize_peak(chunk)
        d = judge.score_detailed(chunk, phys, window_pos=min(max(pos, 0.0), 1.0)); heads.append(d)
    return {k: float(np.mean([h[k] for h in heads])) for k in ('tone_quality', 'overall', 'attack_quality', 'release_quality') if k in heads[0]}

def report(label, rows, pred_key, lab_key):
    sel = [(r['_pred'][pred_key], float(r[lab_key])) for r in rows if r.get('_pred') and r[lab_key] not in ('', 'None')]
    if len(sel) < 5: print(f'  {label}: n={len(sel)} (too few)'); return
    p, l = map(np.array, zip(*sel)); rho, pv = spearmanr(p, l)
    print(f'  {label:42s} n={len(sel):3d}  Spearman {rho:+.3f} (p={pv:.1e})  MAE {np.mean(np.abs(p - l)):.3f} = {np.mean(np.abs(p - l))*3:.2f} rating levels')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--meta', required=True)
    ap.add_argument('--judges', default='both'); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--split-file', default=None, help='Validation split JSON: {"val_group_ids": [...]} or {"val_stroke_ids": [...]} = the checkpoint\'s real validation set')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.manifest)) if r['audio_exists'] == '1' and r.get('flagged', '0') != '1']
    if a.limit: rows = rows[:a.limit]
    recs = {r['stroke_id']: r for r in ds.load_records(a.meta)}
    groups = sorted(set(r['group_id'] for r in rows))
    if a.split_file:
        sp = json.load(open(a.split_file)); val = set(sp.get('val_group_ids', [])) | {r['group_id'] for r in rows if r['stroke_id'] in set(sp.get('val_stroke_ids', []))}
        split_src = f'REAL validation list from {os.path.basename(a.split_file)}'
    else:
        rng = np.random.RandomState(0); rng.shuffle(groups); val = set(groups[:max(1, int(round(len(groups) * 0.2)))])
        split_src = 'REPRODUCED with dataset.group_train_val_split(seed 0, 20 %) on this manifest; exact only if data version, filters and ordering match the training run'
    print(f'{len(rows)} labelled recordings; validation split = {len(val)} of {len(set(groups))} group ids; source: {split_src}')
    json.dump(dict(source=split_src, val_group_ids=sorted(val), train_group_ids=sorted(set(groups) - val),
                   val_stroke_ids=sorted(r['stroke_id'] for r in rows if r['group_id'] in val)), open('split_B_used.json', 'w'), indent=1)
    print('train/validation ids used -> split_B_used.json')
    from rl.piece_env import RealScorer
    for name, ck in JUDGES.items():
        if a.judges != 'both' and a.judges != name: continue
        judge = RealScorer(checkpoint_path=ck); print(f'\n=== judge {name} ({os.path.basename(ck)})')
        for r in rows: r['_pred'] = recording_scores(judge, r['audio_path'], recs[r['stroke_id']])
        for split, sel in (('all labelled (descriptive; mostly seen in training)', rows), ('validation split (out-of-sample if the list is real)', [r for r in rows if r['group_id'] in val])):
            for pk, lk in (('tone_quality', 'tone_quality'), ('overall', 'overall'), ('attack_quality', 'attack_quality'), ('release_quality', 'release_quality')):
                report(f'{split[:44]}: {pk} vs human', sel, pk, lk)
        json.dump({r['stroke_id']: r['_pred'] for r in rows}, open(f'preds_B_{name}.json', 'w'))
    print('\nCaveat: the checkpoints were trained on most of these recordings; only the validation-split rows are out-of-sample, and only if the checkpoint used split seed 0 / 20 % on the same snapshot.')
if __name__ == '__main__': main()
