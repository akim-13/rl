"""Usage: cmp.py BASE OTHER [OTHER...]: paired AUC differences other - base (results/<name>.csv)."""
import csv, sys, numpy as np
R=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "results") + "/"
def auc(n):
    d={}
    for r in csv.DictReader(open(R+n+".csv")):
        if int(float(r["block"]))>=1: d.setdefault(int(r["seed"]),[]).append(float(r["eval_score"]))
    return {s:np.mean(v)/500 for s,v in d.items()}
def ci(x):
    rng=np.random.default_rng(0); x=np.asarray(x); bs=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(10000)]
    return f"{x.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}], better on {(x>0).sum()}/{len(x)}"
base=auc(sys.argv[1]); print(f"{sys.argv[1]} AUC {np.mean(list(base.values())):.3f}")
for o in sys.argv[2:]:
    names=o.split("+"); ms=[auc(n) for n in names]; m={s:np.mean([mm[s] for mm in ms]) for s in ms[0]}
    print(f"{o} AUC {np.mean(list(m.values())):.3f}  minus {sys.argv[1]}: {ci([m[s]-base[s] for s in m])}")
