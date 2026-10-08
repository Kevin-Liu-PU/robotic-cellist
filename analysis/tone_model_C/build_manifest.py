"""Build the C manifest from a dataset folder (metadata jsonl with human annotations + audio dir).
    .venv/bin/python build_manifest.py --meta DIR/metadata_human_current.jsonl --audio-dir DIR/audio --out manifest.csv
                                       [--annotations results/annotations_A1.json results/annotations_A2.json ...]
Labels: per TIER1 field, mean of the annotators' 1-4 ratings mapped to [0,1] (dataset.py's convention); per-annotator
tone columns kept for agreement. Also prints inter-annotator agreement on doubly-rated recordings (part of B).
If the metadata's `annotations` are empty and exports are given, a best-effort merge by stroke_id is attempted and
the export's first item is printed so the field mapping can be fixed. Revision: 2026-09-06.
CELLO_REPO_ROOT defaults to this checkout; data and annotation exports must be
supplied explicitly through the existing CLI options."""
import argparse, json, os, sys, csv, collections
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_paths import repo_root, data_root, require_file, require_directory
REPO = str(repo_root())
DATA_ROOT = str(data_root())
require_file(Path(REPO) / "SoundClassifier" / "audio_features.py", "CELLO_REPO_ROOT")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'SoundClassifier'))
from SoundClassifier import dataset as ds
FIELDS = ds.TIER1_FIELDS   # overall, tone_quality, attack_quality, release_quality, bow_control, dynamic_accuracy

def merge_exports(records, paths):
    by_id = {r['stroke_id']: r for r in records}; by_audio = {r.get('audio_file'): r for r in records}
    for p in paths:
        items = json.load(open(p)); items = items if isinstance(items, list) else items.get('results', items.get('tasks', []))
        print(f'{p}: {len(items)} items; first item keys: {list(items[0].keys()) if items else None}')
        n = 0
        for it in items:
            sid = it.get('stroke_id') or (it.get('data') or {}).get('stroke_id'); af_ = it.get('audio_file') or (it.get('data') or {}).get('audio_file')
            rec = by_id.get(sid) or by_audio.get(af_)
            if rec is None: continue
            ann = {k: it[k] for k in FIELDS if k in it}; ann['annotator'] = it.get('annotator', os.path.basename(p).split('_')[-1].split('.')[0])
            if ann: rec.setdefault('annotations', []).append(ann); n += 1
        print(f'  merged {n} annotations')
    return records

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--meta', required=True); ap.add_argument('--audio-dir', required=True)
    ap.add_argument('--annotations', nargs='*', default=[]); ap.add_argument('--out', required=True); a = ap.parse_args()
    a.meta = str(require_file(a.meta, "--meta"))
    a.audio_dir = str(require_directory(a.audio_dir, "--audio-dir"))
    for filename in a.annotations:
        require_file(filename, "--annotations")
    records = ds.load_records(a.meta)
    if a.annotations and not any(r.get('annotations') for r in records): records = merge_exports(records, a.annotations)
    rows = []; annotators = set()
    for r in records:
        anns = r.get('annotations') or []
        if not anns: continue
        lab = ds.annotation_multidim(r)
        row = dict(stroke_id=r['stroke_id'], audio_path=os.path.join(a.audio_dir, r['audio_file']),
                   audio_exists=int(os.path.exists(os.path.join(a.audio_dir, r['audio_file']))),
                   session_id=r.get('session_id'), condition_label=r.get('condition_label'), condition_type=r.get('condition_type'),
                   config=r.get('config'), string=r.get('string'), stroke_duration=r.get('stroke_duration'),
                   speed_mean=(r.get('measured') or {}).get('speed_mean'), depth_m=(r.get('commanded') or {}).get('depth_m'),
                   n_annot=len(anns), flagged=int(bool(r.get('flagged'))), repeat=r.get('repeat', 0),
                   group_id=f"{r['stroke_id']}_r{r.get('repeat', 0)}")
        for f in FIELDS: row[f] = lab.get(f)
        for ann in anns:
            who = str(ann.get('annotator', '?')); annotators.add(who)
            if 'tone_quality' in ann: row[f'tone_{who}'] = ann['tone_quality']
        rows.append(row)
    cols = list(rows[0].keys()) if rows else []
    for who in sorted(annotators):
        if f'tone_{who}' not in cols: cols.append(f'tone_{who}')
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow(x) for x in rows]
    print(f'manifest: {len(rows)} labelled recordings of {len(records)}; audio present {sum(x["audio_exists"] for x in rows)}; annotators {sorted(annotators)}')
    print('sessions', collections.Counter(x['session_id'] for x in rows).most_common(8))
    print('condition types', collections.Counter(x['condition_type'] for x in rows))
    # inter-annotator agreement on tone (doubly rated)
    from scipy.stats import spearmanr
    whos = sorted(annotators)
    for i in range(len(whos)):
        for j in range(i + 1, len(whos)):
            pairs = [(x[f'tone_{whos[i]}'], x[f'tone_{whos[j]}']) for x in rows if x.get(f'tone_{whos[i]}') is not None and x.get(f'tone_{whos[j]}') is not None]
            if len(pairs) >= 5:
                u, v = zip(*pairs); rho, p = spearmanr(u, v)
                print(f'agreement tone {whos[i]} vs {whos[j]}: n={len(pairs)} Spearman {rho:+.2f} (p={p:.3f}) MAE {np.mean(np.abs(np.array(u)-np.array(v))):.2f} rating levels')
if __name__ == '__main__': main()
