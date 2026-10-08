"""python summarize.py res1.csv res2.csv ... [--mode fixed|auto|all]"""
import sys
import pandas as pd

files = [a for a in sys.argv[1:] if not a.startswith("--")]
mode = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--mode=")), "all")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
if mode == "fixed":
    df = df[df.k_target > 0]
elif mode == "auto":
    df = df[df.k_target == 0]

metrics = ["nmi", "ari", "coh_paper", "coh_strict", "diversity", "outliers", "k_actual", "time"]
pd.set_option("display.width", 200)
print(f"== mode={mode}, runs={len(df)} ==")
print("\n-- overall (mean over datasets, ks, seeds) --")
print(df.groupby("model")[metrics].mean().round(3).sort_values("nmi", ascending=False).to_string())
print("\n-- per dataset: NMI / ARI / coh_strict --")
piv = df.groupby(["dataset", "model"])[["nmi", "ari", "coh_strict", "coh_paper", "diversity", "outliers"]].mean().round(3)
print(piv.to_string())

# head-to-head wins per (dataset, k) on seed-averaged NMI
g = df.groupby(["dataset", "k_target", "model"])["nmi"].mean().unstack("model")
print("\n-- NMI wins per (dataset, k) --")
print(g.idxmax(axis=1).value_counts().to_string())
