"""Summarise runs/*.json: mean +/- SD per configuration and PAIRED differences versus a reference configuration (default M6)."""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from aegis_net.metrics import t_ci, paired_delta
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./results"); ap.add_argument("--ref", default="M6"); a = ap.parse_args()
df = pd.DataFrame([json.load(open(p)) for p in sorted(glob.glob(f"{a.out}/runs/*.json"))])
rows = []
for tag, g in df.groupby("tag"):
    r = dict(tag=tag, n_seeds=g.seed.nunique(), params=int(g.params.iloc[0]))
    for m in ("acc", "weighted_f1", "macro_f1", "macro_f1_major", "macro_f1_minor"):
        mu, lo, hi = t_ci(g[m].values); r[m] = f"{mu:.4f} [{lo:.4f},{hi:.4f}]"; r[m + "_sd"] = round(float(g[m].std(ddof=1)), 4) if len(g) > 1 else None
    if tag != a.ref and a.ref in set(df.tag):
        d = paired_delta(df, "macro_f1", a.ref, tag); r[f"paired d macro_f1 vs {a.ref}"] = f"{d['delta']:+.4f} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]"
    rows.append(r)
os.makedirs(f"{a.out}/tables", exist_ok=True); t = pd.DataFrame(rows); t.to_csv(f"{a.out}/tables/ablation_summary.csv", index=False); print(t.to_string(index=False))
