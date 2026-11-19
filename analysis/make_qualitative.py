"""
Qualitative figure: what margin and influence pick IN THE SAME situation.

Setting fixed BEFORE looking at the images (no hand-picked example):
  Corel5k, ViT-B/16, GCN-Net, 5% budget, split 0, seed 0.
  Cold start = MaxHerding (B/2 = 125 labels, same seeds as exp 06); the GCN is trained once;
  with the SAME model and the SAME labels, margin and influence each select the 1st round (q = 25).
  Shows the first 6 picks of each, in the order they were selected.
  Panel (c): the 1st influence pick and its 6 reverse neighbors with the largest weight x doubt.

Needs the Corel5k images in IMAGES_DIR.
Note: GCN training on a laptop differs slightly from the server; the figure is
illustrative, the numbers of the paper come from the server runs.

Usage (from the repository root):  python3 analysis/make_qualitative.py
Outputs: analysis/figures/qualitative.{pdf,png} and analysis/tables/qualitative_stats.csv
"""

import importlib.util
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data  # noqa: E402
import official_ports as op  # noqa: E402
from active import reverse_matrix, uncertainty_prob  # noqa: E402
from gcn import fit  # noqa: E402
from graphs import build_edge_index  # noqa: E402
from protocol import fixed_test_split  # noqa: E402

_spec = importlib.util.spec_from_file_location("run05", os.path.join(ROOT, "experiments", "05_al_confirm", "run.py"))
run05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run05)

DATASET, DESC, GCN, BUDGET, SPLIT, SEED = "corel5k", "vitb16", "gcn_net", 0.05, 0, 0
SHOW = int(os.environ.get("SHOW", 6))          # imagens por linha (versao compacta: SHOW=5)
COMPACT = os.environ.get("COMPACT") == "1"      # linhas mais juntas (saida qualitative_<SHOW>_compact)
COLUMN = os.environ.get("COLUMN") == "1"        # versao de UMA coluna (PRL curta): saida qualitative_column
IMAGES_DIR = os.environ.get("IMAGES_DIR", "datasets/corel5k/images")
NAMES_FILE = os.path.join(data.DATASETS_DIR, "corel5k-feat", "corel5k_lists.txt")
CACHE = os.path.join(ROOT, "analysis", "qualitative_images")       # so' as miniaturas usadas
ACCENT, WARM, INK, MUTED = "#1f5fbf", "#b8641a", "#1a1a1a", "#6b6b6b"


def thumb(name, size=160):
    os.makedirs(CACHE, exist_ok=True)
    local = os.path.join(CACHE, name)
    if not os.path.exists(local):
        shutil.copyfile(os.path.join(IMAGES_DIR, name), local)
    im = Image.open(local).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size))
    return np.asarray(im)


def main():
    names = [ln.strip() for ln in open(NAMES_FILE) if ln.strip()]
    labels = data.load_labels(DATASET)
    n = len(labels)
    assert len(names) == n
    feats = data.load_features(DATASET, DESC)
    rks = data.load_rks(DATASET, DESC, L=max(run05.GRAPH_K, run05.INFL_K + 1))
    dev = torch.device("cpu")   # CPU: treino deterministico (na GPU o treino varia levemente a cada execucao)
    ei = build_edge_index(rks, run05.GRAPH_K, "rec").to(dev)
    x = torch.from_numpy(feats).to(dev)
    R = reverse_matrix(rks, run05.INFL_K, run05.INFL_P).tocsc()

    pool, test = fixed_test_split(labels, SPLIT)
    B = int(round(BUDGET * n))
    m0 = B // 2
    lab = np.sort(op.maxherding_paper(pool, m0, feats, rng=np.random.default_rng(30_000 + 100 * SPLIT + SEED)))
    logp, _ = fit(GCN, x, ei, labels, lab, SEED, **data.gcn_params(DATASET))
    logp = logp.cpu().numpy() if torch.is_tensor(logp) else logp
    pred = logp.argmax(1)
    u = uncertainty_prob(logp)
    u_nb = u.copy()
    u_nb[lab] = 0
    q = len(np.array_split(np.arange(B - m0), run05.ROUNDS)[0])
    cand = np.setdiff1d(pool, lab)
    picks = {k: np.asarray(run05.query(k, logp, cand, q, np.random.default_rng(20_000 + 100 * SPLIT + SEED),
                                       lab, feats, R.tocsr()))
             for k in ("margin", "influence")}

    def info(v):
        col = R[:, v]
        rev = col.indices
        return {"n_rev": len(rev), "rev_doubt": float(u_nb[rev].mean()) if len(rev) else 0.0,
                "own_doubt": float(u[v]), "wrong": bool(pred[v] != labels[v])}

    rows = []
    for k, pk in picks.items():
        for r, v in enumerate(pk):
            rows.append({"method": k, "order": r + 1, "index": int(v), "image": names[v], **info(v)})
    st = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "analysis", "tables"), exist_ok=True)
    st.to_csv(os.path.join(ROOT, "analysis", "tables", "qualitative_stats.csv"), index=False)
    summ = st.groupby("method")[["n_rev", "rev_doubt", "own_doubt", "wrong"]].mean().round(3)
    print(f"1a rodada: q = {q} escolhas de cada metodo; medias:\n{summ}")
    print("sobreposicao entre as duas listas:", len(set(picks["margin"]) & set(picks["influence"])))

    if COLUMN:
        draw_column(picks, names, info, R, u, u_nb, rks, n=int(os.environ.get("NCOL", "4")))
        return

    # ------------------------------------------------------------ figura
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig = plt.figure(figsize=(7.0 * (SHOW + 0.55) / 6.55, (4.15 if COMPACT else 4.6) * (SHOW + 0.55) / 6.55))
    gs = fig.add_gridspec(3, SHOW + 1, width_ratios=[0.55] + [1] * SHOW, hspace=0.42 if COMPACT else 0.5, wspace=0.08)

    def row(r, key, title, color):
        ax = fig.add_subplot(gs[r, 0])
        ax.axis("off")
        ax.text(1.0, 0.5, title, ha="right", va="center", fontsize=7.5, color=color, fontweight="bold",
                transform=ax.transAxes)
        for j, v in enumerate(picks[key][:SHOW]):
            a = fig.add_subplot(gs[r, j + 1])
            a.imshow(thumb(names[v]))
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color(color)
                s.set_linewidth(1.2)
            a.set_title(["1st", "2nd", "3rd", "4th", "5th", "6th"][j] + " pick", fontsize=6.5, color=MUTED, pad=2)
            d = info(v)
            a.set_xlabel(f"{d['n_rev']} rev. nbrs\nmean doubt {d['rev_doubt']:.2f}", fontsize=6, color=INK,
                         labelpad=2)

    row(0, "margin", "(a) Margin", MUTED)
    row(1, "influence", "(b) Influence", ACCENT)

    # (c) 1a escolha da influencia -> seta -> vizinhos reversos de maior peso x duvida
    #     (NAO e' uma busca: sao as imagens que TEM a escolhida na propria lista top-20)
    v0 = picks["influence"][0]
    col = R[:, v0]
    score = col.data * u_nb[col.indices]
    top = col.indices[np.argsort(-score)][:SHOW - 2]
    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    ax.text(1.0, 0.5, "(c) Why (b)\npicked its\n1st image", ha="right", va="center", fontsize=7.5,
            color=INK, fontweight="bold", transform=ax.transAxes)
    a = fig.add_subplot(gs[2, 1])
    a.imshow(thumb(names[v0]))
    a.set_xticks([])
    a.set_yticks([])
    for sp_ in a.spines.values():
        sp_.set_color(ACCENT)
        sp_.set_linewidth(1.8)
    a.set_title("1st pick of (b)", fontsize=6.5, color=ACCENT, pad=2)
    a.set_xlabel(f"own doubt {u[v0]:.2f}", fontsize=6, color=INK, labelpad=2)
    arr = fig.add_subplot(gs[2, 2])
    arr.axis("off")
    arr.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4))
    arr.text(0.5, 0.68, "its label\nreaches", ha="center", va="bottom", fontsize=6.5, color=ACCENT,
             transform=arr.transAxes)
    arr.text(0.5, 0.32, "doubtful images\nthat rank it\nnear the top", ha="center", va="top", fontsize=6, color=MUTED,
             transform=arr.transAxes)
    for j, i in enumerate(top):
        a = fig.add_subplot(gs[2, j + 3])
        a.imshow(thumb(names[i]))
        a.set_xticks([])
        a.set_yticks([])
        for sp_ in a.spines.values():
            sp_.set_color(MUTED)
            sp_.set_linewidth(0.8)
        pos = int(np.where(rks[i][1:run05.INFL_K + 1] == v0)[0][0]) + 1
        a.set_title("its list: #" + str(pos) + " = pick", fontsize=6, color=MUTED, pad=2)
        a.set_xlabel(f"doubt {u_nb[i]:.2f}", fontsize=6, color=INK, labelpad=2)

    out = os.path.join(ROOT, "analysis", "figures", "qualitative" + ("" if SHOW == 6 and not COMPACT else f"_{SHOW}") + ("_compact" if COMPACT else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    print("figura:", out + ".pdf")


def draw_column(picks, names, info, R, u, u_nb, rks, n=4):
    """Versao de UMA coluna (~8,8 cm): 4 escolhas por linha; titulos das linhas em cima, nao ao lado."""
    plt.rcParams.update({"font.family": "serif", "font.size": 6})
    fig = plt.figure(figsize=(3.45, 3.7 * 4 / n + (0.25 if n > 4 else 0)))  # n=5: miniaturas menores, figura mais baixa
    gs = fig.add_gridspec(3, n, hspace=0.55, wspace=0.13, top=0.93, bottom=0.06, left=0.01, right=0.99)

    def img(ax, name, color, lw):
        ax.imshow(thumb(name))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color(color)
            sp_.set_linewidth(lw)

    def row(r, key, title, color):
        for j, v in enumerate(picks[key][:n]):
            a = fig.add_subplot(gs[r, j])
            img(a, names[v], color, 1.1)
            d = info(v)
            a.set_xlabel(f"{d['n_rev']} rev. nbrs\ndoubt {d['rev_doubt']:.2f}", fontsize=5.4, color=INK, labelpad=1.5)
            if j == 0:
                a.set_title(title, fontsize=6.6, color=color, fontweight="bold", loc="left", pad=3)

    word = {4: "four", 5: "five", 6: "six"}[n]
    row(0, "margin", f"(a) Margin: first {word} picks", MUTED)
    row(1, "influence", f"(b) Influence: first {word} picks", ACCENT)
    v0 = picks["influence"][0]
    col = R[:, v0]
    score = col.data * u_nb[col.indices]
    top = col.indices[np.argsort(-score)][:n - 1]
    a = fig.add_subplot(gs[2, 0])
    img(a, names[v0], ACCENT, 1.7)
    a.set_title("(c) 1st pick of (b) and doubtful images that rank it near the top", fontsize=6.6, color=INK,
                fontweight="bold", loc="left", pad=3)
    a.set_xlabel(f"selected\ndoubt {u[v0]:.2f}", fontsize=5.4, color=ACCENT, labelpad=1.5)
    for j, i in enumerate(top):
        b = fig.add_subplot(gs[2, j + 1])
        img(b, names[i], MUTED, 0.7)
        pos = int(np.where(rks[i][1:run05.INFL_K + 1] == v0)[0][0]) + 1
        b.set_xlabel(f"ranks it #{pos}\ndoubt {u_nb[i]:.2f}", fontsize=5.4, color=INK, labelpad=1.5)
    out = os.path.join(ROOT, "analysis", "figures", "qualitative_column" + ("" if n == 4 else f"_{n}"))
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.01, dpi=250)
    print("figura:", out + ".pdf")


if __name__ == "__main__":
    main()
