"""
Tables that appear only in the supplementary material (the rest of it reuses analysis/tables/*.tex).

  supp_plugin_breakdown.tex  influence - margin per cold start, by budget and by dataset (all 3 lists)
  supp_ablation_budget.tex   ablation and sensitivity by budget (original lists)

Usage (from the repository root): python3 analysis/make_tables.py && python3 analysis/make_supp_tables.py
"""

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
T = os.path.join(ROOT, "analysis", "tables")
LISTS = [("none", "Original"), ("LHRR", "LHRR"), ("UMAP", "UMAP")]
DS = [("flowers", "Flowers"), ("corel5k", "Corel5k"), ("cub200", "CUB")]


def num(x):
    return f"{x + 0.0:+.2f}".replace("-0.00", "+0.00")  # evita "-0.00"


def stars(p):
    return "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def table(cap, label, colspec, header, body, star=False):
    env = "table*" if star else "table"
    return "\n".join(["% gerado por analysis/make_supp_tables.py -- nao editar a mao", f"\\begin{{{env}}}[t]", "\\centering",
                      f"\\caption{{{cap}}}", f"\\label{{{label}}}", "\\footnotesize", f"\\begin{{tabular}}{{{colspec}}}",
                      "\\toprule", *header, "\\midrule", *body, "\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""])


def plugin_breakdown():
    b = pd.read_csv(os.path.join(T, "plugin_breakdown.csv"))
    body = []
    for rr, rname in LISTS:
        body.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{rname} lists}}}} \\\\")
        for cs in ["Random", "Rank-ProbCover", "TypiClust", "MaxHerding"]:
            x = b[(b.cold_start == cs) & (b.lists == rr)]
            cells = []
            for by, vals in [("budget", ["0.02", "0.05", "0.1"]), ("dataset", [d for d, _ in DS])]:
                for v in vals:
                    r = x[(x.by == by) & (x.value.astype(str) == v)].iloc[0]
                    cells.append(f"{num(r.delta)}{stars(r.p)}")
            body.append(f"\\quad {cs} & " + " & ".join(cells) + " \\\\")
    header = ["& \\multicolumn{3}{c}{Budget} & \\multicolumn{3}{c}{Dataset} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              "Cold start & 2\\% & 5\\% & 10\\% & " + " & ".join(n for _, n in DS) + " \\\\"]
    cap = ("Influence query minus margin after the same cold start, by budget (18 configurations each) and by dataset "
           "(18 configurations each). Wilcoxon signed-rank test: $^{*}p<0.05$, $^{**}p<0.01$.")
    with open(os.path.join(T, "supp_plugin_breakdown.tex"), "w") as f:
        f.write(table(cap, "tab:supp-plugin-breakdown", "lrrrrrr", header, body))


def ablation_budget():
    t = pd.read_csv(os.path.join(T, "ablation.csv"))
    t = t[t.lists == "none"]
    names = {"margin": "Margin instead of influence", "A1_nodiscount": "No greedy discount",
             "A2_nopos": "No position weight ($\\rho=1$)", "A3_noself": "No own doubt", "A4_nonbr": "No neighbor doubt",
             "A5_inductive": "Inductive (test doubt ignored)", "S_k10": "$k=10$", "S_k40": "$k=40$",
             "S_p05": "$\\rho=0.5$", "S_p07": "$\\rho=0.7$", "S_p08": "$\\rho=0.8$", "S_r1": "$R=1$", "S_r10": "$R=10$"}
    body = [f"{names[r.variant]} & {num(r.delta_2)} & {num(r.delta_5)} & {num(r.delta_10)} & {num(r.delta)}{stars(r.p)} \\\\"
            for _, r in t.iterrows()]
    header = ["Variant & 2\\% & 5\\% & 10\\% & All \\\\"]
    cap = ("Ablation and sensitivity by budget: accuracy difference to the full method (MaxHerding cold start, "
           "original lists; 6 configurations per budget, 18 in total). The last column repeats the main text, with the "
           "Wilcoxon test over the 18 configurations.")
    with open(os.path.join(T, "supp_ablation_budget.tex"), "w") as f:
        f.write(table(cap, "tab:supp-ablation-budget", "lrrrr", header, body))


if __name__ == "__main__":
    plugin_breakdown()
    ablation_budget()
    print("ok: supp_plugin_breakdown.tex, supp_ablation_budget.tex")
