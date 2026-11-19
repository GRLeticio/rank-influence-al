"""
Generates ALL tables of the paper from the CSV files of the experiments (no number copied by hand).

Inputs: results/05_al_confirm_server, 06_al_baselines_server, 07_ablation_sensitivity_server,
        08_cost_server, 10_tcm_rule_server.
Outputs in analysis/tables/ (each table as .csv for inspection and .tex for the paper):

  means_by_config.csv        mean and std over the 10 splits, per configuration and method (exp 05+06);
                             the source of everything and the full table of the supplementary material
  means_by_config_exp07.csv  same for exp 07
  plugin.{csv,tex}           Table 1: influence - margin after the same cold start, per list type
  plugin_breakdown.csv       same, by budget and by dataset
  sota_<list>.{csv,tex}      Table 2: methods x (dataset, budget), mean over descriptors and GCNs
  sota_<list>_std.{csv,tex}  same with +- standard deviation
  rerank.{csv,tex}           mean accuracy per list type (only configurations present in all three)
  ablation.{csv,tex}         Table 3: ablation and sensitivity (exp 07, exploratory)
  cost.{csv,tex}             selection time (exp 08)

Configuration = dataset x descriptor x list x GCN x budget; value = mean over the 10 splits.
All configurations are included (54 per list type); those used during development are flagged
in means_by_config. Comparisons over configurations: mean difference, wins/losses and paired
Wilcoxon test (Demsar 2006).

Usage (from the repository root): python3 analysis/make_tables.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "analysis", "tables")
CFG = ["dataset", "descriptor", "rerank", "gcn_type", "budget"]
RERANKS = ["none", "LHRR", "UMAP"]
LIST_NAME = {"none": "Original", "LHRR": "LHRR", "UMAP": "UMAP"}
DATASETS = [("flowers", "Flowers-17"), ("corel5k", "Corel5k"), ("cub200", "CUB-200")]
BUDGETS = [0.02, 0.05, 0.10]
PARTIAL_05 = ["probcover", "probcover_margin", "tcm", "typiclust"]   # so' rodaram em parte no exp 05

# Tabela 2: (familia, [(pipeline, nome no artigo)]) -- implementacoes oficiais quando existem
SOTA_ROWS = [
    ("Reference", [("random", "Random"), ("margin", "Margin~\\cite{scheffer2001active}")]),
    ("Coverage (whole budget)", [("probcover_off", "ProbCover~\\cite{yehuda2022active}"),
                                 ("typiclust_off", "TypiClust~\\cite{hacohen2022active}"),
                                 ("maxherding_off", "MaxHerding~\\cite{bae2024generalized}"),
                                 ("rank_probcover", "Rank-ProbCover")]),
    ("Graph AL (whole budget)", [("rim_t0", "RIM~\\cite{zhang2021rim}"),
                                 ("rim_tmed", "RIM (median $\\theta$)~\\cite{zhang2021rim}"),
                                 ("grain", "Grain~\\cite{zhang2021grain}")]),
    ("Random start + query", [("coreset_off", "CoreSet~\\cite{sener2017active}"),
                              ("info_density", "Information density~\\cite{settles2008analysis}"),
                              ("age", "AGE~\\cite{cai2017active}")]),
    ("Coverage + query", [("tcm_off", "TCM ($B/2$ switch)~\\cite{doucet2024bridging}"), ("dcom", "DCoM~\\cite{mishal2024dcom}"),
                          ("maxherding_off_margin", "MaxHerding + margin"),
                          ("rank_probcover_margin", "Rank-ProbCover + margin")]),
    ("Influence (ours)", [("random_influence", "Random + influence"), ("ours", "Rank-ProbCover + influence"),
                          ("typiclust_off_influence", "TypiClust + influence"),
                          ("maxherding_off_influence", "MaxHerding + influence")]),
    # bloco proprio para a regra de troca dos autores do TCM (decidida depois dos resultados principais);
    # continua nos ranks e testes, so' muda a posicao na tabela
    ("Switch rule of the TCM authors (post hoc)", [("tcm_rule1", "TCM (authors' rule)~\\cite{doucet2024bridging}"),
                          ("typiclust_rule1_influence", "TypiClust + influence (authors' rule)")]),
]

# Tabela 1: (peca 1, com influencia, com margem)
PLUGIN = [
    ("Random", "random_influence", "margin"),
    ("Rank-ProbCover", "ours", "rank_probcover_margin"),
    ("TypiClust", "typiclust_off_influence", "tcm_off"),
    ("MaxHerding", "maxherding_off_influence", "maxherding_off_margin"),
]

RERANK_METHODS = ["margin", "tcm_off", "maxherding_off", "maxherding_off_margin", "dcom", "rim_tmed",
                  "ours", "maxherding_off_influence"]

ABLATION = [  # (variante, nome no artigo, grupo)
    ("margin", "Margin instead of influence", "Ablation"),
    ("A1_nodiscount", "No greedy discount", "Ablation"),
    ("A2_nopos", "No position weight ($\\rho=1$)", "Ablation"),
    ("A3_noself", "No own doubt", "Ablation"),
    ("A4_nonbr", "No neighbor doubt", "Ablation"),
    ("A5_inductive", "Inductive (test doubt ignored)", "Ablation"),
    ("S_k10", "$k=10$", "Sensitivity"), ("S_k40", "$k=40$", "Sensitivity"),
    ("S_p05", "$\\rho=0.5$", "Sensitivity"), ("S_p07", "$\\rho=0.7$", "Sensitivity"),
    ("S_p08", "$\\rho=0.8$", "Sensitivity"),
    ("S_r1", "$R=1$ round", "Sensitivity"), ("S_r10", "$R=10$ rounds", "Sensitivity"),
]


# ------------------------------------------------------------------ utilitarios

def load(name):
    return pd.read_csv(os.path.join(ROOT, "results", name, "results.csv"))


def per_config(df):
    g = df.groupby(CFG + ["pipeline"]).acc
    m = g.agg(["mean", "std", "count"]).reset_index()
    m.columns = CFG + ["pipeline", "acc_mean", "acc_std", "n_splits"]
    return m


def seen_before(df):
    """Configuracoes 'vistas antes' (PREREGISTRO.md, desvios): Flowers/ResNet152 em todas as listas
    (usado na exploracao) e Flowers/ViT-B/16 com UMAP (previa local de 2026-10-01)."""
    fl = df.dataset == "flowers"
    return fl & ((df.descriptor == "Resnet152") | ((df.descriptor == "vitb16") & (df.rerank == "UMAP")))


def confirmatory(df):
    """Decisao de 2026-10-06: o artigo usa TODAS as configuracoes (54 por lista), sem excluir as vistas
    durante o desenvolvimento; a coluna 'seen_during_development' do means_by_config permite a conferencia
    de robustez (as conclusoes nao mudam sem elas)."""
    return df


def paired(d):
    d = d.dropna()
    p = wilcoxon(d).pvalue if (d != 0).any() and len(d) > 1 else np.nan
    return {"delta": d.mean(), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": len(d), "p": p}


def stars(p):
    return "" if np.isnan(p) else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def write(df, name, tex=None):
    df.to_csv(os.path.join(OUT, name + ".csv"), index=False, float_format="%.4f")
    if tex is not None:
        with open(os.path.join(OUT, name + ".tex"), "w") as f:
            f.write(tex)
    print(f"  {name}")


def tex_table(caption, label, colspec, header, body, star=False, note=None, colsep=None):
    env = "table*" if star else "table"
    lines = [f"% gerado por analysis/make_tables.py -- nao editar a mao",
             f"\\begin{{{env}}}[t]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize"]
    if colsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{colsep}}}")
    # encolhe so' se passar da largura (precisa de \usepackage{adjustbox}); table* usa \textwidth, que no
    # suplementar (article, 1 coluna) e' menor que no artigo
    width = "\\textwidth" if star else "\\columnwidth"
    lines += [f"\\begin{{adjustbox}}{{max width={width}}}",
              f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *header, "\\midrule", *body, "\\bottomrule",
              "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines += [f"\\\\[2pt]{{\\scriptsize {note}}}"]
    lines += [f"\\end{{{env}}}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ tabelas

def table_means(df05_06, df07):
    m = per_config(df05_06)
    m["seen_during_development"] = seen_before(m)
    write(m, "means_by_config")
    write(per_config(df07), "means_by_config_exp07")


def table_plugin(cfg):
    rows, brk = [], []
    for rr in RERANKS:
        c = cfg.xs(rr, level="rerank")
        for name, a, b in PLUGIN:
            d = c[a] - c[b]
            rows.append({"cold_start": name, "lists": rr, "acc_margin": c[b].mean(), "acc_influence": c[a].mean(),
                         **paired(d)})
            for bb in BUDGETS:
                brk.append({"cold_start": name, "lists": rr, "by": "budget", "value": bb,
                            **paired(d.xs(bb, level="budget"))})
            for ds, _ in DATASETS:
                brk.append({"cold_start": name, "lists": rr, "by": "dataset", "value": ds,
                            **paired(d.xs(ds, level="dataset"))})
    t = pd.DataFrame(rows)
    write(pd.DataFrame(brk), "plugin_breakdown")

    body = []
    for name, _, _ in PLUGIN:
        cells = []
        for rr in RERANKS:
            r = t[(t.cold_start == name) & (t.lists == rr)].iloc[0]
            cells.append(f"{r.delta:+.2f}{stars(r.p)} & {r.wins}/{r.losses}")
        body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    ns = {rr: int(t[t.lists == rr].n.iloc[0]) for rr in RERANKS}
    header = ["& \\multicolumn{2}{c}{Original} & \\multicolumn{2}{c}{LHRR} & \\multicolumn{2}{c}{UMAP} \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
              "Cold start & $\\Delta$ & W/L & $\\Delta$ & W/L & $\\Delta$ & W/L \\\\"]
    cap = ("Influence query vs.\\ margin after the same cold start: mean accuracy difference (points) over "
           f"configurations (Original: {ns['none']}, LHRR: {ns['LHRR']}, UMAP: {ns['UMAP']}) and "
           "wins/losses. Wilcoxon signed-rank test: $^{*}p<0.05$, $^{**}p<0.01$.")
    write(t, "plugin", tex_table(cap, "tab:plugin", "lrrrrrr", header, body))


def table_sota(cfg, sd_cfg):
    for rr in RERANKS:
        c = cfg.xs(rr, level="rerank")
        # metodos que so' rodaram em algumas listas (ex.: exp 10 parte 2, so' listas originais) ficam de fora das outras
        pipes = [p for _, fam in SOTA_ROWS for p, _ in fam if p in c and c[p].notna().any()]
        c = c[pipes].dropna()
        # desvio: desvio padrao nas 10 divisoes de cada configuracao, medio sobre as configuracoes da celula
        sd = sd_cfg.xs(rr, level="rerank")[pipes].loc[c.index]
        sdm = sd.groupby(level=["dataset", "budget"]).mean().T
        cols = [(ds, b) for ds, _ in DATASETS for b in BUDGETS]
        mean = c.groupby(level=["dataset", "budget"]).mean().T
        rank = c.rank(axis=1, ascending=False).mean()
        t = pd.DataFrame({f"{ds}_{int(b*100)}": mean[(ds, b)] for ds, b in cols})
        t["avg_acc"], t["avg_rank"] = c.mean(), rank
        t.insert(0, "method", t.index)
        # negrito pelo valor IMPRESSO: empates visiveis ficam todos em negrito.
        # Avg. com 2 casas (o texto discute diferencas como 0,26 e 0,36); as celulas ficam com 1.
        dig = {k: (2 if k == "avg_acc" else 1) for k in t.columns}
        best = {k: t[k].round(dig[k]).max() for k in t.columns if k not in ("method", "avg_rank")}
        best["avg_rank"] = t["avg_rank"].round(1).min()

        def cell(k, v):
            s = f"{v:.{dig[k]}f}"
            return f"\\textbf{{{s}}}" if np.isclose(round(v, dig[k]), best[k]) else s
        sdt = pd.DataFrame({f"{ds}_{int(b*100)}": sdm[(ds, b)] for ds, b in cols})
        sdt["avg_acc"] = sd.mean()

        # versao curta: em vez do diagrama, marcar quem NAO difere do melhor por rank medio
        # (Wilcoxon par a par sobre todos os pares + Holm -- o mesmo teste do diagrama, analysis/make_cd.py)
        import itertools
        from scipy.stats import wilcoxon as _wil
        top = rank.idxmin()
        pairs = list(itertools.combinations(pipes, 2))
        pv = np.array([_wil(c[a], c[b]).pvalue if (c[a] != c[b]).any() else 1.0 for a, b in pairs])
        order = np.argsort(pv)
        adj = np.empty(len(pv))
        adj[order] = np.minimum(1, np.maximum.accumulate((len(pv) - np.arange(len(pv))) * pv[order]))
        sig = {frozenset(pr): q < 0.05 for pr, q in zip(pairs, adj)}
        tied = {p for p in pipes if p != top and not sig[frozenset((top, p))]}

        def rows(with_sd, mark=False):
            body = []
            for fam, items in SOTA_ROWS:
                body.append(f"\\multicolumn{{12}}{{l}}{{\\textit{{{fam}}}}} \\\\")
                for p, nice in items:
                    if p not in t.index:
                        continue
                    r = t.loc[p]
                    cells = []
                    for k in t.columns[1:]:
                        x = cell(k, r[k])
                        if with_sd and k != "avg_rank":
                            x += f"{{\\scriptsize$\\pm${sdt.loc[p, k]:.1f}}}"
                        cells.append(x)
                    name = nice
                    if mark and p == top:
                        name = nice.replace("~\\cite", "$^{*}$~\\cite") if "~\\cite" in nice else nice + "$^{*}$"
                    elif mark and p in tied:
                        name = nice.replace("~\\cite", "$^{\\dagger}$~\\cite") if "~\\cite" in nice else nice + "$^{\\dagger}$"
                    body.append(f"\\quad {name} & " + " & ".join(cells) + " \\\\")
            return body
        body = rows(False)
        header = ["& \\multicolumn{3}{c}{Flowers-17} & \\multicolumn{3}{c}{Corel5k} & \\multicolumn{3}{c}{CUB-200} & & \\\\",
                  "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
                  "Method & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in ["2\\%", "5\\%", "10\\%"] * 3 + ["Avg.", "Rank"]) + " \\\\"]
        n = len(c)
        cap = (f"Accuracy (\\%) with {LIST_NAME[rr].lower() if rr == 'none' else LIST_NAME[rr]} ranked lists, "
               f"averaged over descriptors and GCNs ({n} configurations). Avg.: mean accuracy; "
               "Rank: mean rank over configurations (lower is better); best in bold. The last block (switch rule "
               "of the TCM authors) was chosen after the main results.")
        write(t.reset_index(drop=True), f"sota_{rr}",
              tex_table(cap, f"tab:sota-{rr.lower()}", "l" + "r" * 11, header, body, star=True))
        cap_sd = cap.replace("best in bold.", "best in bold. $\\pm$: std.\\ over the 10 splits, "
                                              "averaged over each cell.")
        tsd = sdt.add_prefix("sd_").reset_index(drop=True)
        write(pd.concat([t.reset_index(drop=True), tsd], axis=1), f"sota_{rr}_std",
              tex_table(cap_sd, f"tab:sota-{rr.lower()}-std", "l" + "r" * 11, header, rows(True), star=True,
                        colsep="3.5pt"))
        cap_mark = cap_sd + (" $^{*}$: best mean rank; $^{\\dagger}$: not significantly different from it "
                             "(pairwise Wilcoxon tests, Holm correction, $\\alpha=0.05$); every other "
                             "method is significantly worse.")
        write(pd.DataFrame({"method": pipes, "best": [p == top for p in pipes], "tied_with_best": [p in tied for p in pipes]}),
              f"sota_{rr}_std_mark",
              tex_table(cap_mark, f"tab:sota-{rr.lower()}-std", "l" + "r" * 11, header, rows(True, mark=True),
                        star=True, colsep="3.5pt"))


def table_rerank(cfg):
    common = None
    for rr in RERANKS:
        idx = cfg.xs(rr, level="rerank")[RERANK_METHODS].dropna().index
        common = idx if common is None else common.intersection(idx)
    rows = []
    for p in RERANK_METHODS:
        r = {"method": p}
        for rr in RERANKS:
            r[rr] = cfg.xs(rr, level="rerank").loc[common, p].mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    t["n_configs"] = len(common)
    nice = {p: n for _, fam in SOTA_ROWS for p, n in fam}   # inclui o \\cite
    body = [f"{nice.get(r.method, r.method)} & " + " & ".join(f"{r[rr]:.2f}" for rr in RERANKS) +
            f" & {r.LHRR - r['none']:+.2f} & {r.UMAP - r['none']:+.2f} \\\\" for _, r in t.iterrows()]
    header = ["Method & Original & LHRR & UMAP & $\\Delta_{\\mathrm{LHRR}}$ & $\\Delta_{\\mathrm{UMAP}}$ \\\\"]
    cap = (f"Mean accuracy (\\%) with the original, LHRR and UMAP ranked lists, over the {len(common)} configurations "
           "available for all three types of list.")
    write(t, "rerank", tex_table(cap, "tab:rerank", "lrrrrr", header, body))


def table_ablation(df07):
    cfg = df07.groupby(CFG + ["pipeline"]).acc.mean().unstack("pipeline")
    rows = []
    for rr in ["none", "LHRR"]:
        c = cfg.xs(rr, level="rerank")
        for v, nice, grp in ABLATION:
            if v not in c or c[v].isna().all():
                continue
            d = c[v] - c["full"]
            r = {"variant": v, "name": nice, "group": grp, "lists": rr, **paired(d)}
            for bb in BUDGETS:
                r[f"delta_{int(bb*100)}"] = d.xs(bb, level="budget").mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    body, last = [], None
    for v, nice, grp in ABLATION:
        if grp != last:
            if last is not None:
                body.append("\\midrule")
            last = grp
        o = t[(t.variant == v) & (t.lists == "none")].iloc[0]
        lh = t[(t.variant == v) & (t.lists == "LHRR")]
        sg = lambda x: f"{x + 0.0:+.2f}".replace("-0.00", "+0.00")  # evita "-0.00"
        lcell = f"{sg(lh.iloc[0].delta)}{stars(lh.iloc[0].p)}" if len(lh) else "--"
        body.append(f"{nice} & {sg(o.delta)}{stars(o.p)} & {o.wins}/{o.losses} & {lcell} \\\\")
    header = ["Variant & $\\Delta$ & W/L & $\\Delta$ (LHRR) \\\\"]
    n = int(t[t.lists == "none"].n.iloc[0])
    cap = ("Ablation and sensitivity (exploratory): accuracy difference to the full method "
           f"($k=20$, $\\rho=0.9$, $R=5$) over {n} configurations (3 datasets $\\times$ ResNet-152, ViT-B/16 "
           "$\\times$ GCN $\\times$ 3 budgets), with MaxHerding as cold start. "
           "$^{*}p<0.05$, $^{**}p<0.01$ (Wilcoxon).")
    write(t, "ablation", tex_table(cap, "tab:ablation", "lrrr", header, body))
    # versao curta (PRL): so' a ablacao; a sensibilidade fica no texto da 6.5 e na Tabela S6 do suplementar
    short = body[:body.index("\\midrule")] if "\\midrule" in body else body
    cap_s = ("Ablation (exploratory): accuracy difference to the full method "
             f"over {n} configurations (3 datasets $\\times$ ResNet-152, ViT-B/16 $\\times$ GCN $\\times$ 3 budgets), "
             "with MaxHerding as cold start; W/L: configurations in which the variant is better/worse. "
             "$^{*}p<0.05$, $^{**}p<0.01$ (Wilcoxon).")
    with open(os.path.join(OUT, "ablation_short.tex"), "w") as f:
        f.write(tex_table(cap_s, "tab:ablation", "lrrr", header, short))


def table_cost():
    """Tabela de custo (exp 08, servidor): tempo so' da escolha, mediana de 3 repeticoes, orcamento de 10%."""
    path = os.path.join(ROOT, "results", "08_cost_server", "results.csv")
    if not os.path.exists(path):
        return
    c = pd.read_csv(path)
    cold = [("sorteio", "Random"), ("Rank-ProbCover (nosso)", "Rank-ProbCover"),
            ("ProbCover (oficial)", "ProbCover~\\cite{yehuda2022active}"), ("DCoM (inicial)", "DCoM~\\cite{mishal2024dcom}"),
            ("MaxHerding (fiel)", "MaxHerding~\\cite{bae2024generalized}"), ("RIM", "RIM~\\cite{zhang2021rim}"),
            ("Grain", "Grain~\\cite{zhang2021grain}"),
            ("TypiClust (oficial)", "TypiClust~\\cite{hacohen2022active}")]
    query = [("margem", "Margin~\\cite{scheffer2001active}"), ("CoreSet (oficial, embeddings)", "CoreSet~\\cite{sener2017active}"),
             ("influencia por rank (nossa)", "Influence (ours)"),
             ("information density", "Information density~\\cite{settles2008analysis}"),
             ("DCoM (rodada, com deltas)", "DCoM~\\cite{mishal2024dcom}"), ("AGE (oficial)", "AGE~\\cite{cai2017active}")]
    ref = [("(referencia) 1 treino da GCN-Net", "One GCN training")]
    dss = [("corel5k", "Corel5k"), ("cub200", "CUB-200")]
    get = lambda ds, m: float(c[(c.dataset == ds) & (c.metodo == m)].segundos.iloc[0])
    n = {ds: (int(c[(c.dataset == ds) & (c.peca == 1)].n_escolhas.iloc[0]),
              int(c[(c.dataset == ds) & (c.peca == 2)].n_escolhas.iloc[0])) for ds, _ in dss}
    rows, body = [], []
    for grp, items, k in [("Cold start", cold, 0), ("One query round", query, 1), ("Reference", ref, None)]:
        hdr = grp if k is None else f"{grp} ({n['corel5k'][k]} / {n['cub200'][k]} picks)"
        body.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{hdr}}}}} \\\\")
        for m, nice in items:
            vals = [get(ds, m) for ds, _ in dss]
            rows.append({"group": grp, "method": nice.split("~")[0], **{ds: v for (ds, _), v in zip(dss, vals)}})
            cell = lambda v: "$<$0.01" if v < 0.005 else f"{v:.2f}"
            name = f"\\textbf{{{nice}}}" if "ours" in nice else nice
            body.append(f"\\quad {name} & " + " & ".join(cell(v) for v in vals) + " \\\\")
    header = ["Method & Corel5k & CUB-200 \\\\"]
    cap = ("Selection time in seconds with a 10\\% budget (median of three runs, one NVIDIA RTX 2080 Ti), "
           "excluding the training of the GCN, shown for reference.")
    write(pd.DataFrame(rows), "cost", tex_table(cap, "tab:cost", "lrr", header, body))


def completeness(df):
    print("\nconferencia (configuracoes por lista, exp 05+06):")
    for rr in RERANKS:
        n = df[df.rerank == rr].groupby(["dataset", "descriptor", "gcn_type", "budget"]).ngroups
        print(f"  {rr:5s}: {n} de 54" + ("" if n == 54 else "   <-- INCOMPLETO"))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.concat([load("05_al_confirm_server"), load("06_al_baselines_server"), load("10_tcm_rule_server")],
                   ignore_index=True)
    df = df[~df.pipeline.isin(PARTIAL_05)]
    df07 = load("07_ablation_sensitivity_server")
    print(f"gerando em {OUT}:")
    table_means(df, df07)
    conf = confirmatory(df)
    cfg = conf.groupby(CFG + ["pipeline"]).acc.mean().unstack("pipeline")
    table_plugin(cfg)
    sd_cfg = conf.groupby(CFG + ["pipeline"]).acc.std().unstack("pipeline")
    table_sota(cfg, sd_cfg)
    table_rerank(cfg)
    table_ablation(df07)
    table_cost()
    completeness(conf)


if __name__ == "__main__":
    main()
