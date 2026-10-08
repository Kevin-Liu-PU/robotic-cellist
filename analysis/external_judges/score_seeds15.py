"""External judges on all 5 pieces x 3 seeds (15 seeds, 90 takes); reuses cache + score_takes judges. Also collects internal posthoc deltas."""
import sys, json, time, glob, os, statistics as st
from pathlib import Path
sys.argv=[sys.argv[0]]; import score_takes as S
CP=S.CP; P=S.P
MAN={
 "twinkle_s0":[(P("20260901_203157"),P("20260901_203332")),(P("20260901_203418"),P("20260901_203552")),(P("20260901_203638"),P("20260901_203813"))],
 "twinkle_s1":S.pairs_from(S.parse_log("ab_twinkle_alt2_s1.log")), "twinkle_s2":S.pairs_from(S.parse_log("ab_twinkle_alt2_s2.log")),
 "maysong_s0":S.pairs_from(S.parse_log("ab_maysong_alt2_s0_20260902.log")), "maysong_s1":S.pairs_from(S.parse_log("ab_maysong_m0-11_alt2_s1.log")), "maysong_s2":S.pairs_from(S.parse_log("ab_maysong_m0-11_alt2_s2.log")),
 "bach_s0":S.pairs_from(S.parse_log("ab_bach_sarabande_m0-7_alt2_s0.log")), "bach_s1":S.pairs_from(S.parse_log("ab_bach_sarabande_m0-7_alt2_s1.log")), "bach_s2":S.pairs_from(S.parse_log("ab_bach_sarabande_m0-7_alt2_s2.log")),
 "vocalise_s0":S.pairs_from(S.parse_log("ab_vocalise_alt2_s0.log")), "vocalise_s1":S.pairs_from(S.parse_log("ab_vocalise_alt2_s1.log")), "vocalise_s2":S.pairs_from(S.parse_log("ab_vocalise_alt2_s2.log")),
 "popper_s0":S.pairs_from(S.parse_log("ab_yunpiece_m9-24_alt2_s0.log")), "popper_s1":S.pairs_from(S.parse_log("ab_yunpiece_m9-24_alt2_s1.log")), "popper_s2":S.pairs_from(S.parse_log("ab_yunpiece_m9-24_alt2_s2.log")),
}
for k,v in MAN.items(): assert len(v)==3,(k,len(v))
scores={}
for f in sorted(glob.glob(str(S.OUT/"scores_*.json"))): scores.update(json.load(open(f)).get("scores",{}))
dirs=sorted({d for v in MAN.values() for pr in v for d in pr}); todo=[d for d in dirs if d.name not in scores or "PAM" not in scores[d.name]]
print(f"15 seeds, {len(dirs)} takes; cached {len(dirs)-len(todo)}, to score {len(todo)}",flush=True)
if todo:
    ab=S.load_audiobox()
    for d in todo: scores[d.name]={k:float(v) for k,v in ab.forward([{"path":S.wav_of(d)}])[0].items()}
    pam,DS=S.load_pam(); ds=DS([S.wav_of(d) for d in todo])
    for i,d in enumerate(todo):
        f,a,idx=ds.collate([ds[i]]); s,seg=pam.evaluate(a,idx); scores[d.name]["PAM"]=float(s[0]); scores[d.name]["PAM_chunks"]=[float(v) for v in seg[0]]
# internal posthoc deltas
def post(d,j): 
    p=f"{d}/posthoc_{j}.json"; jj=json.load(open(p)); return jj["mean_tone"], jj["mean_overall"]
out={}
for k,v in MAN.items():
    r={"train_tone":[],"held_tone":[],"held_overall":[],"PAM":[],"PQ":[],"CE":[],"PC":[]}
    for b,rl in v:
        r["train_tone"].append(post(rl,"multilength")[0]-post(b,"multilength")[0])
        ht=post(rl,"human_A1_A5"); hb=post(b,"human_A1_A5"); r["held_tone"].append(ht[0]-hb[0]); r["held_overall"].append(ht[1]-hb[1])
        for x in ("PAM","PQ","CE","PC"): r[x].append(scores[rl.name][x]-scores[b.name][x])
    out[k]={m:{"mean":st.mean(vals),"pairs":vals} for m,vals in r.items()}
json.dump({"seeds":out},open(S.OUT/f"deltas_15seeds_{time.strftime('%Y%m%d_%H%M')}.json","w"),indent=1)
json.dump({"scores":{d.name:scores[d.name] for d in dirs}},open(S.OUT/f"scores_seeds15_{time.strftime('%Y%m%d_%H%M')}.json","w"),indent=1)
print(f"{'seed':12s} train_tone held_tone held_over    PAM     PQ     CE     PC")
for k,r in out.items(): print(f"{k:12s} {r['train_tone']['mean']:+9.3f} {r['held_tone']['mean']:+9.3f} {r['held_overall']['mean']:+9.3f} {r['PAM']['mean']:+6.3f} {r['PQ']['mean']:+6.3f} {r['CE']['mean']:+6.3f} {r['PC']['mean']:+6.3f}")
