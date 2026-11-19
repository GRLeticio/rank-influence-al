"""
Summarizes experiment 07 (exploratory): each variant against the full influence ("full")
and against margin, paired by split; test over configurations (Demsar 2006) and per dataset.

Usage: python3 experiments/07_ablation_sensitivity/summarize.py [results.csv]
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "07_ablation_sensitivity", "results.csv")
df = pd.read_csv(path)
CFG = ["dataset", "descriptor", "rerank", "gcn_type", "budget"]
u = df.groupby(CFG + ["split", "pipeline"], as_index=False).acc.mean()
pd.set_option("display.width", 220)

print("# Acuracia media (%) por variante")
print(u.groupby(["rerank", "pipeline", "dataset"]).acc.mean().unstack("dataset").round(2).to_string())


def compare(a, b, rerank):
    rows = []
    for k, g in u[u.rerank == rerank].groupby(CFG):
        p = g.pivot(index="split", columns="pipeline", values="acc")
        if a in p and b in p:
            d = (p[a] - p[b]).dropna()
            if len(d):
                rows.append(dict(zip(CFG, k), d=d.mean()))
    r = pd.DataFrame(rows)
    if r.empty:
        return None
    p_all = wilcoxon(r.d).pvalue if len(r) >= 6 and np.any(r.d != 0) else np.nan
    per = " | ".join(f"{ds}: {g.d.mean():+.2f} ({(g.d > 0).sum()}/{len(g)})" for ds, g in r.groupby("dataset"))
    return f"{r.d.mean():+.2f} ({(r.d > 0).sum()}/{len(r)}, p={p_all:.3f}) | {per}"


for rr in sorted(u.rerank.unique()):
    print(f"\n# lista: {rr}. Diferenca em pontos (variante - referencia); configuracoes com ganho / total")
    for v in sorted(set(u[u.rerank == rr].pipeline) - {"full"}):
        print(f"  {v:15s} vs full:   {compare(v, 'full', rr)}")
    if "margin" in set(u[u.rerank == rr].pipeline):
        print(f"  {'full':15s} vs margin: {compare('full', 'margin', rr)}")
