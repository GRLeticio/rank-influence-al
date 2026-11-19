"""
Analysis of experiment 06 together with 05.

Joins both results.csv files (same unit key) and tests H5-H7:
  H5  ours vs tcm_off
  H6  ours vs the best (by mean, in each configuration) of AGE / info density / RIM / Grain / DCoM
  H7  typiclust_off_influence vs tcm_off
Unit = split; paired Wilcoxon; Holm within each hypothesis.

Usage: python3 experiments/06_al_baselines/summarize.py [results05.csv] [results06.csv] [--full]
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
args = [x for x in sys.argv[1:] if not x.startswith("--")]
p05 = args[0] if len(args) > 0 else os.path.join(ROOT, "results", "05_al_confirm", "results.csv")
p06 = args[1] if len(args) > 1 else os.path.join(ROOT, "results", "06_al_baselines", "results.csv")
df = pd.concat([pd.read_csv(p05), pd.read_csv(p06)], ignore_index=True)
pd.set_option("display.width", 240, "display.max_rows", 2000)

CFG = ["dataset", "descriptor", "rerank", "gcn_type", "budget"]
GRAPH_AL = ["age", "info_density", "rim_t0", "rim_tmed", "grain", "dcom"]
unit = df.groupby(CFG + ["split", "pipeline"], as_index=False).acc.mean()


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def compare(name, a_name, rival_fn):
    rows = []
    for keys, g in unit.groupby(CFG):
        piv = g.pivot(index="split", columns="pipeline", values="acc")
        if a_name not in piv:
            continue
        rival = rival_fn(piv)
        if rival is None or rival not in piv:
            continue
        d = (piv[a_name] - piv[rival]).dropna()
        if len(d) == 0:
            continue
        p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan
        rows.append(dict(zip(CFG, keys), rival=rival, delta=d.mean(), wins=(d > 0).mean(), n=len(d), p=p))
    res = pd.DataFrame(rows)
    if res.empty:
        print(f"\n# {name}: sem configuracoes completas ainda")
        return
    res["p_holm"] = holm(res.p.values)
    res["sig"] = np.where(res.p_holm < 0.05, np.where(res.delta > 0, "ganho", "perda"), "")
    print(f"\n# {name}  ({len(res)} configuracoes; delta em pontos)")
    for by in ("dataset", "budget"):
        print(res.groupby(by).agg(configs=("delta", "size"), delta_medio=("delta", "mean"),
                                  vitorias=("wins", "mean"),
                                  ganho_sig=("sig", lambda s: (s == "ganho").sum()),
                                  perda_sig=("sig", lambda s: (s == "perda").sum())).round(3).to_string())
    if name.startswith("H6"):
        print("rival escolhido:", res.rival.value_counts().to_dict())
    if "--full" in sys.argv:
        print(res.sort_values(CFG).to_string(index=False, float_format="%.3f"))


print("# Acuracia media (%) por pipeline (dataset x orcamento)")
t = unit.groupby(["dataset", "budget", "pipeline"]).acc.mean().unstack(["dataset", "budget"])
print(t.sort_values(t.columns[-1], ascending=False).round(2).to_string())

compare("H5: ours vs tcm_off", "ours", lambda piv: "tcm_off")
compare("H6: ours vs melhor AL em grafo", "ours",
        lambda piv: max([c for c in GRAPH_AL if c in piv], key=lambda c: piv[c].mean(), default=None))
compare("H7: typiclust_off_influence vs tcm_off", "typiclust_off_influence", lambda piv: "tcm_off")
