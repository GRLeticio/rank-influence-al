"""
Analysis of exp 09: joins 1/3/7% (exp 09) with 2/5/10% (exp 06) and draws the budget curves.

  1. check: the 5% job of exp 09 must match exp 06 EXACTLY (same code and seeds)
  2. table: difference to pure MaxHerding per dataset x budget (mean over descriptors and GCNs)
  3. figure: x axis = labels per class; y axis = accuracy - pure MaxHerding
     -> analysis/figures/budget_curves.{pdf,png} and analysis/tables/budget_curves.csv

Usage (from the repository root):  python3 experiments/09_budget_curves/summarize.py
  (looks for results/<exp> and, if it does not exist, results/<exp>_server)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPES = ["maxherding_off", "maxherding_off_margin", "maxherding_off_influence",
         "typiclust_off", "tcm_off", "typiclust_off_influence"]
CLASSES = {"corel5k": 50, "cub200": 200}
NAMES = {"corel5k": "Corel5k (50 classes)", "cub200": "CUB-200 (200 classes)"}
CFG = ["dataset", "descriptor", "gcn_type", "budget"]
STYLE = {  # (rotulo, cor, estilo, marcador) -- azul = com a nossa influencia
    "maxherding_off_influence": ("MaxHerding + influence", "#1f5fbf", "-", "o"),
    "typiclust_off_influence": ("TypiClust + influence", "#86aee6", "--", "s"),
    "maxherding_off_margin": ("MaxHerding + margin", "#b8641a", "-", "o"),
    "tcm_off": ("TCM (B/2 switch)", "#e0a46b", "--", "s"),
    "typiclust_off": ("TypiClust (whole budget)", "#6b6b6b", "--", "s"),
}


def find(name):
    for d in (name, name + "_server"):
        p = os.path.join(ROOT, "results", d, "results.csv")
        if os.path.exists(p):
            return p
    return None


def main():
    p06, p09, pchk = find("06_al_baselines"), find("09_budget_curves"), find("09_budget_curves_check")
    e06 = pd.read_csv(p06)
    e06 = e06[(e06.rerank == "none") & e06.dataset.isin(CLASSES) & e06.pipeline.isin(PIPES)]

    if pchk:
        # o treino na GPU nao e' bit a bit deterministico: compara a MEDIA nas 10 divisoes, por metodo
        chk = pd.read_csv(pchk).set_index("key")
        ref = e06.set_index("key").reindex(chk.index)
        both = pd.DataFrame({"pipeline": chk.pipeline, "exp09": chk.acc, "exp06": ref.acc})
        m = both.groupby("pipeline")[["exp09", "exp06"]].mean()
        m["diff"] = m.exp09 - m.exp06
        ok = (m["diff"].abs() < 0.5).all() and len(chk) == 60
        print(f"conferencia (Corel5k/ViT/GCN-Net, 5%, {len(chk)} linhas de 60): media exp 09 vs exp 06")
        print(m.round(2).to_string())
        print("  OK: pode juntar" if ok else "  <-- DIFERENCA >= 0,5 ponto ou incompleto: investigar antes de juntar")
    else:
        print("conferencia: results/09_budget_curves_check nao encontrado")

    parts = [e06]
    if p09:
        parts.append(pd.read_csv(p09))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.pipeline.isin(PIPES) & df.dataset.isin(CLASSES)]
    cfg = df.groupby(CFG + ["pipeline"]).agg(acc=("acc", "mean"), cov=("coverage", "mean")).reset_index()
    piv = cfg.pivot_table(index=CFG, columns="pipeline", values="acc")
    full = piv.dropna()
    diff = full.sub(full["maxherding_off"], axis=0).drop(columns="maxherding_off")
    g = diff.groupby(level=["dataset", "budget"])
    tab = g.mean().round(2)
    tab["n_configs"] = g.size()
    wins = (diff["maxherding_off_influence"] > 0).groupby(level=["dataset", "budget"]).mean().round(2)
    tab["influence_wins_frac"] = wins
    cov = cfg.pivot_table(index=["dataset", "budget"], columns="pipeline", values="cov")
    tab["cov_mh_pure"] = cov["maxherding_off"].round(3)
    tab["cov_mh_infl"] = cov["maxherding_off_influence"].round(3)
    tab = tab.reset_index()
    tab.insert(2, "labels_per_class", [b * {"corel5k": 5000, "cub200": 11788}[d] / CLASSES[d]
                                       for d, b in zip(tab.dataset, tab.budget)])
    os.makedirs(os.path.join(ROOT, "analysis", "tables"), exist_ok=True)
    tab.to_csv(os.path.join(ROOT, "analysis", "tables", "budget_curves.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nacuracia - MaxHerding puro (media sobre descritor e GCN):")
    print(tab.to_string(index=False))

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    h_in = float(os.environ.get("FIG_H", "2.5"))  # FIG_H=1.9 -> versao mais baixa (PRL curta), salva como budget_curves_short
    fig, axes = plt.subplots(1, 2, figsize=(7.0, h_in), sharey=False)
    for ax, ds in zip(axes, CLASSES):
        t = tab[tab.dataset == ds].sort_values("budget")
        x = t.labels_per_class.values
        ax.axhline(0, color="#1a1a1a", lw=0.8)
        ax.text(x.max(), 0, " MaxHerding\n (whole budget)", va="center", ha="left", fontsize=6.5, color="#1a1a1a")
        for p, (lab, col, ls, mk) in STYLE.items():
            ax.plot(x, t[p].values, color=col, ls=ls, marker=mk, ms=3.5, lw=1.6 if col in ("#1f5fbf", "#b8641a") else 1.0,
                    label=lab)
        ax.set_title(NAMES[ds], fontsize=8)
        ax.set_xlabel("labels per class (budget / number of classes)")
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.1f}".rstrip("0").rstrip(".") for v in x])
    axes[0].set_ylabel("accuracy difference (points)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    out = os.path.join(ROOT, "analysis", "figures", "budget_curves" + ("_short" if "FIG_H" in os.environ else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    print(f"\nfigura: {out}.pdf")


if __name__ == "__main__":
    main()
