"""Markdown benchmark tables from h2h CSVs: python gen_bench_md.py out.md res1.csv res2.csv ..."""
import sys
import pandas as pd

out, files = sys.argv[1], sys.argv[2:]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
NAMES = {"tritopic[v2.4]": "**TriTopic 2.4**", "tritopic[base2.3]": "TriTopic 2.3", "bertopic": "BERTopic (default)",
         "bertopic_tuned": "BERTopic (tuned)"}
ORDER = [m for m in NAMES if m in set(df.model)]
DS = {"20newsgroups": "20 Newsgroups", "bbc_news": "BBC News", "agnews": "AG News", "arxiv": "arXiv"}


def table(d, cols, fmt):
    g = d.groupby("model")[list(cols)].mean().reindex(ORDER)
    lines = ["| Model | " + " | ".join(cols.values()) + " |", "|---" * (len(cols) + 1) + "|"]
    for m, row in g.iterrows():
        lines.append(f"| {NAMES[m]} | " + " | ".join(fmt.get(c, "{:.3f}").format(row[c]) for c in cols) + " |")
    return "\n".join(lines)


cols = {"nmi": "NMI", "ari": "ARI", "coh_strict": "NPMI (strict)", "coh_paper": "NPMI (2.3 benchmark)",
        "diversity": "Diversity", "outliers": "Outliers", "time": "Fit time (s)"}
fmt = {"outliers": "{:.1%}", "time": "{:.1f}", "diversity": "{:.2f}"}
fixed, auto = df[df.k_target > 0], df[df.k_target == 0]

md = ["### Fixed topic count (k from the paper grid)", "", table(fixed, cols, fmt), "",
      "### Automatic topic count (`n_topics=\"auto\"` / BERTopic default)", "",
      table(auto, {**cols, "k_actual": "Topics found"}, {**fmt, "k_actual": "{:.1f}"}), "",
      "### NMI per dataset (fixed k)", ""]
piv = fixed.pivot_table(index="dataset", columns="model", values="nmi", aggfunc="mean")[ORDER]
md.append("| Dataset | " + " | ".join(NAMES[m] for m in ORDER) + " |")
md.append("|---" * (len(ORDER) + 1) + "|")
for ds, row in piv.iterrows():
    best = row.idxmax()
    md.append(f"| {DS.get(ds, ds)} | " + " | ".join(
        (f"**{row[m]:.3f}**" if m == best else f"{row[m]:.3f}") for m in ORDER) + " |")

stab = fixed.groupby(["model", "dataset", "k_target"]).nmi.agg(lambda x: x.max() - x.min()).groupby("model").agg(["mean", "max"]).reindex(ORDER)
md += ["", "### Seed stability (fixed k): NMI spread across 3 seeds", "",
       "| Model | Mean spread | Worst spread |", "|---|---|---|"]
md += [f"| {NAMES[m]} | {r['mean']:.3f} | {r['max']:.3f} |" for m, r in stab.iterrows()]
vs = fixed[fixed.model != "tritopic[base2.3]"]
wins = vs.groupby(["dataset", "k_target", "model"]).nmi.mean().unstack().idxmax(axis=1).value_counts()
md += ["", f"Against BERTopic (default and tuned), TriTopic 2.4 has the better NMI in **{wins.get('tritopic[v2.4]', 0)} of {wins.sum()}** dataset/k combinations."]
open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
