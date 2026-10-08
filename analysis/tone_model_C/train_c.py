"""Train, validate (nested grouped CV) and freeze the audio-only tone model C on the human-labelled recordings.
    .venv/bin/python train_c.py --manifest manifest.csv --meta DATA/metadata_human_current.jsonl \
                                --target tone_quality --out model_C_YYYYMMDD.joblib [--outer 5 --inner 4]
Nested cross-validation (review 9/6): the OUTER loop holds out whole recording sessions (GroupKFold) and is
the only source of the reported Spearman / MAE; the INNER loop (GroupKFold on the outer training part) picks
the model and hyper-parameters; imputation and standardisation live inside each candidate pipeline, so they
are fit on training rows only. After validation, the same inner selection runs once on ALL labelled data,
the chosen pipeline is refit on all data and frozen with the outer metrics attached. The 45 policy pairs and
the D takes are never read here. Fold counts are set once the session count is known (defaults 5 / 4;
both are capped by the number of sessions). """
import argparse, csv, hashlib, json, os, sys, time
import numpy as np, joblib
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import features_c as F

CANDIDATES = {
    'ridge_a1':   lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
    'ridge_a10':  lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
    'ridge_a100': lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=100.0)),
    'ridge_a1000': lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1000.0)),
    'hgb_d3':     lambda: make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=10, random_state=0)),
    'hgb_d2':     lambda: make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=2, min_samples_leaf=15, random_state=0)),
    'rf':         lambda: make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=-1)),
}

def load_features(rows, cache):
    X = []
    for r in rows:
        key = hashlib.md5((r['audio_path'] + F.VERSION).encode()).hexdigest(); p = os.path.join(cache, key + '.npy')
        if os.path.exists(p): X.append(np.load(p)); continue
        y, sr = F.recording_segment(r['audio_path'], r.get('_record')); v = F.segment_features(y, sr); np.save(p, v); X.append(v)
    return np.array(X, dtype=np.float64)

def select_inner(X, y, groups, n_inner):
    """Inner grouped CV on the given rows only: returns (best_name, {name: spearman})."""
    k = max(2, min(n_inner, len(set(groups)))); gkf = GroupKFold(n_splits=k); scores = {}
    for name, mk in CANDIDATES.items():
        oof = np.zeros_like(y)
        for tr, te in gkf.split(X, y, groups):
            m = mk(); m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
        scores[name] = float(spearmanr(oof, y)[0])
    return max(scores, key=scores.get), scores

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--target', default='tone_quality')
    ap.add_argument('--out', required=True); ap.add_argument('--cache', default='feature_cache'); ap.add_argument('--meta', default=None)
    ap.add_argument('--outer', type=int, default=5); ap.add_argument('--inner', type=int, default=4); a = ap.parse_args(); os.makedirs(a.cache, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.manifest)) if r['audio_exists'] == '1' and r[a.target] not in ('', 'None') and r.get('flagged', '0') != '1']
    if a.meta:
        sys.path.insert(0, F.REPO); sys.path.insert(0, os.path.join(F.REPO, 'SoundClassifier')); from SoundClassifier import dataset as ds
        recs = {r['stroke_id']: r for r in ds.load_records(a.meta)}
        for r in rows: r['_record'] = recs.get(r['stroke_id'])
    y = np.array([float(r[a.target]) for r in rows]); groups = np.array([r['session_id'] for r in rows]); n_sess = len(set(groups))
    print(f'{len(rows)} labelled recordings, {n_sess} sessions, target {a.target}: mean {y.mean():.3f} sd {y.std():.3f}')
    X = load_features(rows, a.cache)
    k_outer = max(2, min(a.outer, n_sess)); outer = GroupKFold(n_splits=k_outer); oof = np.zeros_like(y); picks = []; fold_rows = []
    for i, (tr, te) in enumerate(outer.split(X, y, groups)):
        best, inner_scores = select_inner(X[tr], y[tr], groups[tr], a.inner)
        m = CANDIDATES[best](); m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te]); picks.append(best)
        rho = float(spearmanr(oof[te], y[te])[0]) if len(te) > 4 else float('nan'); mae = float(np.mean(np.abs(oof[te] - y[te])))
        fold_rows.append(dict(fold=i, n_test=int(len(te)), sessions_test=sorted(set(groups[te])), picked=best, inner_spearman=inner_scores[best], outer_spearman=rho, outer_mae=mae))
        print(f'  outer fold {i}: test n={len(te)} sessions={len(set(groups[te]))}  inner pick {best} ({inner_scores[best]:+.3f})  outer Spearman {rho:+.3f}  MAE {mae:.3f}')
    rho_all, p_all = spearmanr(oof, y); mae_all = float(np.mean(np.abs(oof - y)))
    print(f'NESTED (report this): pooled outer Spearman {rho_all:+.3f} (p={p_all:.1e})  MAE {mae_all:.3f} = {mae_all*3:.2f} rating levels  picks {picks}')
    # final procedure on all data: same inner selection, then refit and freeze
    best_all, scores_all = select_inner(X, y, groups, a.inner); final = CANDIDATES[best_all](); final.fit(X, y)
    print(f'final model (selected on all labelled data by the same inner procedure): {best_all}  inner scores {json.dumps({k: round(v, 3) for k, v in scores_all.items()})}')
    meta = dict(model=best_all, target=a.target, n=len(rows), n_sessions=n_sess, protocol='nested GroupKFold by session_id',
                outer_folds=k_outer, inner_folds=max(2, min(a.inner, n_sess)), nested=dict(spearman=float(rho_all), p=float(p_all), mae=mae_all, mae_levels=mae_all * 3.0, picks=picks, folds=fold_rows),
                inner_scores_all=scores_all, report={best_all: dict(spearman=float(rho_all), mae=mae_all)},
                feature_version=F.VERSION, feature_names=F.feature_names(), manifest=os.path.abspath(a.manifest),
                manifest_md5=hashlib.md5(open(a.manifest, 'rb').read()).hexdigest(), frozen_at=time.strftime('%Y-%m-%d %H:%M'))
    joblib.dump(dict(model=final, meta=meta), a.out); json.dump(meta, open(a.out + '.json', 'w'), indent=1)
    np.save(a.out + '.oof.npy', np.stack([oof, y])); print('frozen ->', a.out)
if __name__ == '__main__': main()
