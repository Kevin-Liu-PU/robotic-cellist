"""Zero-lineage external judges (Audiobox PQ/CE/CU/PC, PAM) on the 3-piece x 3-seed live A/Bs.
Reuses score_takes.py judges + caches already-scored takes. Revision: 2026-09-03."""
import sys, json, time, glob, statistics as st
from pathlib import Path
sys.argv = [sys.argv[0]]                       # score_takes parses argv at import? (it doesn't, but be safe)
import score_takes as S
CP = S.CP; P = S.P
MAN = {
 "twinkle_s0": [(P("20260901_203157"), P("20260901_203332")), (P("20260901_203418"), P("20260901_203552")), (P("20260901_203638"), P("20260901_203813"))],
 "twinkle_s1": S.pairs_from(S.parse_log("ab_twinkle_alt2_s1.log")),
 "twinkle_s2": S.pairs_from(S.parse_log("ab_twinkle_alt2_s2.log")),
 "maysong_s0": S.pairs_from(S.parse_log("ab_maysong_alt2_s0_20260902.log")),
 "maysong_s1": S.pairs_from(S.parse_log("ab_maysong_m0-11_alt2_s1.log")),
 "maysong_s2": S.pairs_from(S.parse_log("ab_maysong_m0-11_alt2_s2.log")),
 "bach_s0":    S.pairs_from(S.parse_log("ab_bach_sarabande_m0-7_alt2_s0.log")),
 "bach_s1":    S.pairs_from(S.parse_log("ab_bach_sarabande_m0-7_alt2_s1.log")),
 "bach_s2":    S.pairs_from(S.parse_log("ab_bach_sarabande_m0-7_alt2_s2.log")),
}
for k,v in MAN.items(): assert len(v)==3, (k, len(v))
# cache: any earlier results file
scores = {}
for f in sorted(glob.glob(str(S.OUT / "scores_*.json"))):
    scores.update(json.load(open(f)).get("scores", {}))
dirs = sorted({d for v in MAN.values() for pr in v for d in pr})
todo = [d for d in dirs if d.name not in scores or "PAM" not in scores[d.name]]
print(f"9 seeds, {len(dirs)} takes; cached {len(dirs)-len(todo)}, to score {len(todo)}", flush=True)
if todo:
    t=time.time(); ab=S.load_audiobox()
    for d in todo:
        o=ab.forward([{"path": S.wav_of(d)}])[0]; scores[d.name]={k: float(v) for k,v in o.items()}
    print(f"audiobox {time.time()-t:.0f}s", flush=True)
    t=time.time(); pam,DS=S.load_pam(); ds=DS([S.wav_of(d) for d in todo])
    for i,d in enumerate(todo):
        f,audios,idx=ds.collate([ds[i]]); s,seg=pam.evaluate(audios,idx)
        scores[d.name]["PAM"]=float(s[0]); scores[d.name]["PAM_chunks"]=[float(v) for v in seg[0]]
    print(f"PAM {time.time()-t:.0f}s", flush=True)
stamp=time.strftime("%Y%m%d_%H%M")
(S.OUT/f"scores_seeds9_{stamp}.json").write_text(json.dumps({"scores":{d.name:scores[d.name] for d in dirs},"manifest":{k:[(b.name,r.name) for b,r in v] for k,v in MAN.items()}},indent=1))
AX=["PAM","PQ","CE","CU","PC"]
print(f"\n{'seed':11s}"+"".join(f"{x:>22s}" for x in AX)+"   (paired RL-Base: mean, n_pos/3)")
allrows={}
for k,v in MAN.items():
    row=[]
    for x in AX:
        ds_=[scores[r.name][x]-scores[b.name][x] for b,r in v]; row.append((st.mean(ds_), sum(z>0 for z in ds_)))
    allrows[k]=row
    print(f"{k:11s}"+"".join(f"{m:+9.3f} ({p}/3)      " for m,p in row))
print("\nBase absolute levels (mean over the 3 Base takes):")
for k,v in MAN.items():
    print(f"  {k:11s}"+"  ".join(f"{x}={st.mean(scores[b.name][x] for b,_ in v):.3f}" for x in AX))
print("\nPer-piece pooled (9 pairs):")
for piece in ("twinkle","maysong","bach"):
    ks=[k for k in MAN if k.startswith(piece)]
    for x in AX:
        ds_=[scores[r.name][x]-scores[b.name][x] for k in ks for b,r in MAN[k]]
        print(f"  {piece:8s} {x:3s} {st.mean(ds_):+.3f} (SD {st.stdev(ds_):.3f}) {sum(z>0 for z in ds_)}/9 positive")
