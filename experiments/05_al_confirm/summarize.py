"""
Analysis of experiment 05.

Statistical unit = test split (mean over seeds, if more than 1).
Configuration = dataset x descriptor x list x GCN x budget.

  1. table of mean accuracy per pipeline (all configurations)
  2. H1-H3: ours vs tcm / maxherding_margin / rank_probcover_margin, per configuration:
     mean delta, wins, Wilcoxon; Holm within each hypothesis;
     number of configurations with significant gains/losses, per dataset
  3. H4: accuracy with the initial half only, Rank-ProbCover vs ProbCover and MaxHerding

Usage: python3 experiments/05_al_confirm/summarize.py [path/results.csv] [--full]
       --full prints the table of each configuration (long)
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
args = [x for x in sys.argv[1:] if not x.startswith("--")]
path = args[0] if args else os.path.join(ROOT, "results", "05_al_confirm", "results.csv")
df = pd.read_csv(path)
pd.set_option("display.width", 240, "display.max_rows", 2000)

CFG = ["dataset", "descriptor", "rerank", "gcn_type", "budget"]
HYP = {"H1": "tcm", "H2": "maxherding_margin", "H3": "rank_probcover_margin"}

unit = df.groupby(CFG + ["split", "pipeline"], as_index=False)[["acc", "acc_initial"]].mean()


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


print("# Acuracia media (%) por pipeline, media sobre todas as configuracoes de cada dataset x orcamento")
t = unit.groupby(["dataset", "budget", "pipeline"]).acc.mean().unstack(["dataset", "budget"])
print(t.sort_values(t.columns[-1], ascending=False).round(2).to_string())

if "--full" in sys.argv:
    print("\n# Por configuracao")
    print(unit.groupby(CFG + ["pipeline"]).acc.mean().unstack("pipeline").round(2).to_string())

for h, rival in HYP.items():
    rows = []
    for keys, g in unit.groupby(CFG):
        a = g[g.pipeline == "ours"].set_index("split").acc
        b = g[g.pipeline == rival].set_index("split").acc
        d = (a - b.reindex(a.index)).dropna()
        if len(d) == 0:
            continue
        p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan
        rows.append(dict(zip(CFG, keys), delta=d.mean(), wins=(d > 0).mean(), n=len(d), p=p))
    res = pd.DataFrame(rows)
    if res.empty:
        continue
    res["p_holm"] = holm(res.p.values)
    res["sig"] = np.where(res.p_holm < 0.05, np.where(res.delta > 0, "ganho", "perda"), "")
    print(f"\n# {h}: ours vs {rival}  (delta em pontos; Holm sobre as {len(res)} configuracoes)")
    print(res.groupby("dataset").agg(configs=("delta", "size"), delta_medio=("delta", "mean"),
                                     ganho_sig=("sig", lambda s: (s == "ganho").sum()),
                                     perda_sig=("sig", lambda s: (s == "perda").sum())).round(3).to_string())
    print(res.groupby("budget").agg(delta_medio=("delta", "mean"),
                                    ganho_sig=("sig", lambda s: (s == "ganho").sum()),
                                    perda_sig=("sig", lambda s: (s == "perda").sum())).round(3).to_string())
    if "--full" in sys.argv:
        print(res.sort_values(CFG).to_string(index=False, float_format="%.3f"))

print("\n# H4: acuracia so' com a metade inicial (sem consultas)")
init = unit[unit.pipeline.isin(["ours", "probcover_margin", "maxherding_margin", "tcm", "margin"])]
h4 = init.groupby(["dataset", "budget", "pipeline"]).acc_initial.mean().unstack("pipeline")
h4 = h4.rename(columns={"ours": "rank_probcover", "probcover_margin": "probcover",
                        "maxherding_margin": "maxherding", "tcm": "typiclust", "margin": "random"})
print(h4.round(2).to_string())
