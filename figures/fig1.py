"""Figure 1 (overview).  python fig1.py  ->  fig1_lists.pdf (used in the paper), fig1_graph.pdf (+ png)"""
import sys
import numpy as np
from figkit import *

W = 7.0                      # largura final (pol.) -- conferir \the\textwidth no Overleaf

# ---------------------------------------------------------------- miniaturas (Flowers-17)
SUN = [721, 725, 730, 741, 750, 760]
DAI = [801, 805, 812, 820, 830, 835, 840, 845, 850, 860]
PAN = [1281, 1285, 1290, 1300, 1310]
V, VP = 801, 645                     # v (candidata influente) e v′ (duvidosa isolada)


# ================================================================= (a) pipeline + laco
def draw_a(ax, y0):
    panel_title(ax, 0.05, y0 + 2.24, "(a) Pipeline and active learning loop")
    cy = y0 + 1.48                                  # linha de cima
    # imagens
    s = 0.40
    for n, i in enumerate([1281, 801, 721]):
        thumb(ax, i, 0.12 + n * 0.07, cy - s / 2 - 0.07 + n * 0.07, s, z=3 + 2 * n)   # cada uma na sua camada
    text(ax, 0.39, cy - 0.40, r"Images $\mathcal{X}$")
    arrow(ax, (0.75, cy), (0.93, cy))
    # extrator
    ax.add_patch(Polygon([[0.95, cy - 0.27], [1.55, cy - 0.15], [1.55, cy + 0.15], [0.95, cy + 0.27]], closed=True,
                         fc=LIGHT, ec="k", lw=0.6, zorder=2))
    text(ax, 1.25, cy, "pre-trained\nCNN / ViT")
    text(ax, 1.25, cy - 0.40, "features")
    arrow(ax, (1.57, cy), (1.75, cy))
    # listas ranqueadas
    ts, g = 0.16, 0.025
    rows = [[721, 725, 730, 741, 750], [801, 805, 812, 820, 830], [1281, 1285, 1290, 1300, 1310]]
    lx = 1.78
    for r, row in enumerate(rows):
        yy = cy + 0.13 - r * (ts + 0.045) - ts / 2 + 0.05
        thumb(ax, row[0], lx, yy, ts, lw=0.6)
        for j, i in enumerate(row[1:]):
            thumb(ax, i, lx + 0.23 + j * (ts + g), yy, ts)
    text(ax, lx + 0.23 + 2 * (ts + g) - g / 2, cy + 0.33, r"$1\ \ 2\ \ \cdots\ \ k$", fs=FS - 1)
    text(ax, lx + 0.47, cy - 0.40, r"ranked lists $\tau_i$")
    # re-ranking opcional (laco tracejado sobre as listas)
    arrow(ax, (lx + 0.95, cy + 0.28), (lx + 0.10, cy + 0.28), ls=(0, (3, 2)), rad=0.55, lw=0.6)
    text(ax, lx + 1.02, cy + 0.50, "optional re-ranking\n(LHRR, UMAP)", fs=FS - 1, style="italic", ha="left")
    arrow(ax, (2.90, cy), (3.10, cy))
    # grafo reciproco
    rng = np.random.default_rng(3)
    cl = [(3.38, cy + 0.10), (3.80, cy + 0.14), (3.55, cy - 0.18)]
    pts = []
    for c, (cx_, cy_) in enumerate(cl):
        for m in range(4):
            a = 2 * np.pi * m / 4 + c
            pts.append((cx_ + 0.11 * np.cos(a), cy_ + 0.09 * np.sin(a), c))
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if pts[a][2] == pts[b][2] and (b - a) in (1, 2):
                ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="0.4", lw=0.5, zorder=3)
    ax.plot([pts[1][0], pts[8][0]], [pts[1][1], pts[8][1]], color="0.4", lw=0.5, zorder=3)
    ax.plot([pts[6][0], pts[10][0]], [pts[6][1], pts[10][1]], color="0.4", lw=0.5, zorder=3)
    for n, (px, py, c) in enumerate(pts):
        if n in (0, 5, 9):                                       # rotulados: quadrado preto
            ax.add_patch(Rectangle((px - 0.03, py - 0.03), 0.06, 0.06, fc="k", ec="k", lw=0.4, zorder=4))
        else:
            ax.add_patch(Circle((px, py), 0.032, fc="white", ec="k", lw=0.5, zorder=4))
    text(ax, 3.58, cy - 0.40, r"reciprocal $k$NN graph")
    arrow(ax, (4.00, cy), (4.22, cy))
    # GCN
    box(ax, 4.24, cy - 0.25, 0.62, 0.50, fc=LIGHT)
    for c, n in enumerate([4, 3, 2]):
        for m in range(n):
            ax.add_patch(Circle((4.36 + c * 0.19, cy + 0.06 + (m - (n - 1) / 2) * 0.09), 0.025, fc="white", ec="k",
                                lw=0.4, zorder=4))
    text(ax, 4.55, cy - 0.17, "GCN")
    arrow(ax, (4.88, cy), (5.06, cy))
    # probabilidades p_i
    for j, h in enumerate([0.30, 0.20, 0.06, 0.03]):
        ax.add_patch(Rectangle((5.10 + j * 0.07, cy - 0.15), 0.05, h, fc="0.35", ec="none", zorder=3))
    ax.plot([5.08, 5.41], [cy - 0.15, cy - 0.15], color="k", lw=0.4)
    text(ax, 5.26, cy - 0.40, r"$p_i$")

    # ---- linha de baixo: cold start -> L -> (GCN) ; consulta -> oraculo -> L
    by = y0 + 0.56
    box(ax, 0.10, by - 0.24, 2.40, 0.48,
        "Cold start (no model): $\\lfloor B/2\\rfloor$ labels\nby a coverage method (plug-in, e.g., MaxHerding)")
    arrow(ax, (1.25, cy - 0.50), (1.25, by + 0.25), ls=(0, (3, 2)), lw=0.6)
    # conjunto rotulado L
    lx0 = 4.14
    box(ax, lx0, by - 0.17, 0.42, 0.34, fc=LIGHT)
    text(ax, lx0 + 0.21, by + 0.04, r"$\mathcal{L}$")
    for m in range(3):
        ax.add_patch(Rectangle((lx0 + 0.09 + m * 0.09, by - 0.12), 0.055, 0.055, fc="k", zorder=4))
    arrow(ax, (2.51, by), (lx0 - 0.02, by))
    text(ax, 3.30, by + 0.10, r"$\lfloor B/2\rfloor$ labels", fs=FS - 1)
    arrow(ax, (lx0 + 0.21, by + 0.18), (lx0 + 0.21, cy - 0.26))
    text(ax, lx0 + 0.25, by + 0.36, "train", ha="left", fs=FS - 1)
    # consulta (contribuicao)
    qx0, qx1 = 4.80, 6.12
    box(ax, qx0, by - 0.27, qx1 - qx0, 0.54,
        "Rank-based influence\nquery (ours) $\\rightarrow$ panel (b)", ec=ACC, lw=1.6)
    arrow(ax, (5.26, cy - 0.50), (5.26, by + 0.28))
    text(ax, 5.30, by + 0.37, r"$u(i)$", ha="left", fs=FS - 1)
    arrow(ax, (2.90, cy - 0.46), (qx0 - 0.01, by + 0.13), ls=(0, (3, 2)), lw=0.6)
    text(ax, 3.62, by + 0.27, r"$\mathcal{N}_k(i)$", fs=FS - 1)
    # oraculo
    arrow(ax, (qx1 + 0.01, by), (6.27, by))
    text(ax, 6.19, by + 0.10, r"$q$", fs=FS - 1)
    box(ax, 6.28, by - 0.24, 0.66, 0.48, "oracle\nlabels")
    # volta: oraculo -> L
    yb = by - 0.37
    ax.plot([6.61, 6.61, lx0 + 0.21], [by - 0.25, yb, yb], color="k", lw=0.7, zorder=4)
    arrow(ax, (lx0 + 0.21, yb), (lx0 + 0.21, by - 0.18))
    text(ax, 5.40, yb - 0.09, r"add $q$ labels, retrain; $R=5$ rounds", fs=FS - 1)


# ================================================================= (b) consulta
U = {"i1": 0.80, "i2": 0.60, "i3": 0.70, "i4": 0.12, "v": 0.70, "vp": 0.95}
RHO_ILL = 0.6        # so' para o desenho do desconto (sem numeros na figura)
ROWS = [("i1", 812, [V, 820, 830, 840], 0), ("i2", 820, [835, V, 812, 845], 1),
        ("i3", 830, [812, 850, V, 860], 2), ("i4", 725, [730, 741, 750, 760], None)]
S, PITCH = 0.27, 0.36


def b_header(ax, y0):
    panel_title(ax, 0.05, y0 + 2.56, "(b) Rank-based influence query (one round)")
    text(ax, 0.10, y0 + 0.02, r"bar $=$ doubt $u(i)=1-(p_i^{(1)}-p_i^{(2)})$ ($u=0$ if labeled);"
         r"  hatched $=$ doubt removed by the discount",
         ha="left", va="bottom", fs=FS - 1)


def b1_lists(ax, y0):
    """vizinhos reversos lidos nas listas: as celulas onde v aparece."""
    x_g, x_q, x_l = 0.12, 0.22, 0.66
    top = y0 + 1.72
    text(ax, x_q + S / 2, top + S + 0.12, r"image $i$", fs=FS - 1)
    text(ax, x_l + 1.5 * (S + 0.05), top + S + 0.12, r"its list $\mathcal{N}_k(i)$:  pos $1,2,3,\dots$", fs=FS - 1)
    for r, (key, qi, lst, pos) in enumerate(ROWS):
        y = top - r * PITCH
        gauge(ax, x_g, y, S, U[key])
        thumb(ax, qi, x_q, y, S, lw=0.6)
        text(ax, x_q + S + 0.09, y + S / 2, r"$\rightarrow$")
        for j, i in enumerate(lst):
            thumb(ax, i, x_l + j * (S + 0.05), y, S, alpha=1.0 if j == pos else 0.5)
            if j == pos:
                frame(ax, x_l + j * (S + 0.05), y, S)
        wx = x_l + 4 * (S + 0.05) + 0.08
        if pos is not None:
            text(ax, wx, y + S / 2, r"$w(i\!\to\! v)=\rho^{%d}$" % pos, ha="left", fs=FS - 1)
        else:
            text(ax, wx, y + S / 2, r"$v\notin\mathcal{N}_k(i)$", ha="left", fs=FS - 1, color=GRAY)
    bx = 2.62
    yb0, yb1 = top - 2 * PITCH, top + S
    ax.plot([bx, bx + 0.05, bx + 0.05, bx], [yb1, yb1, yb0, yb0], color=ACC, lw=0.9)
    text(ax, bx + 0.10, (yb0 + yb1) / 2, "reverse\nneighbors\nof $v$", ha="left", fs=FS - 1, color=ACC)
    y = top - 4 * PITCH - 0.06
    gauge(ax, x_g, y, S, U["vp"])
    thumb(ax, VP, x_q, y, S, lw=0.6)
    text(ax, x_q + S + 0.07, y + S / 2, r"$v′$: very doubtful, but in no list  $\Rightarrow\ I(v′)=0$", ha="left",
         fs=FS - 1)


def b1_graph(ax, y0):
    """vizinhos reversos como setas i -> v (espessura ~ w), v′ sem setas."""
    s = 0.30
    vx, vy = 1.55, y0 + 1.10                         # v no centro
    nb = [("i1", 812, (0.55, y0 + 1.70), 0), ("i2", 820, (0.40, y0 + 0.80), 1),
          ("i3", 830, (1.45, y0 + 0.30), 2)]
    others = [("i4", 725, (2.55, y0 + 1.75), None)]
    text(ax, 1.30, y0 + 2.18, r"arrow $i\to v$: $v$ is in the list of $i$ (thicker = higher)", fs=FS - 1)

    def node(key, img, x, y, sel=False, right=False):
        thumb(ax, img, x - s / 2, y - s / 2, s, lw=0.6)
        gauge(ax, x + s / 2 + 0.05 if right else x - s / 2 - 0.10, y - s / 2, s, U[key])
        if sel:
            frame(ax, x - s / 2, y - s / 2, s)

    node("v", V, vx, vy, sel=True, right=True)
    text(ax, vx, vy + s / 2 + 0.10, "$v$", fs=FS)
    for key, img, (x, y), pos in nb:
        node(key, img, x, y)
        lw = [2.4, 1.5, 0.8][pos]
        d = np.array([vx - x, vy - y]); d /= np.linalg.norm(d)
        arrow(ax, (x + d[0] * 0.22, y + d[1] * 0.22), (vx - d[0] * 0.24, vy - d[1] * 0.24), lw=lw, color=ACC, head=7)
        mx, my = (x + vx) / 2, (y + vy) / 2
        text(ax, mx + 0.05, my + 0.09, r"$\rho^{%d}$" % pos, fs=FS - 1, color=ACC, ha="left")
    # vizinhos que v nao recebe (outra regiao, imagens certas)
    for key, img, (x, y), _ in others:
        node(key, img, x, y)
        node("i4", 730, x + 0.05, y - 0.55)
        arrow(ax, (x, y - s / 2 - 0.02), (x + 0.05, y - 0.55 + s / 2 + 0.02), lw=0.6, color=GRAY)
    # v′ isolada
    px, py = 2.65, y0 + 0.50
    node("vp", VP, px, py)
    text(ax, px + s / 2 + 0.06, py + s / 2 - 0.02, "$v′$", ha="left", fs=FS)
    text(ax, px, py - s / 2 - 0.13, "no incoming arrows\n$\\Rightarrow I(v′)=0$", fs=FS - 1)
    text(ax, 0.45, y0 + 0.42, "reverse\nneighbors of $v$", fs=FS - 1, color=ACC)


def b_compare(ax, y0, cx=3.45):
    top = y0 + 1.72
    text(ax, cx + 0.62, top + S + 0.12, "who to label?", fs=FS - 1, weight="bold")
    text(ax, cx + 0.62, top + S - 0.08, r"$I(v)=\sum_{i:\,v\in\mathcal{N}_k(i)} w(i\!\to\! v)\,u(i)$", fs=FS - 1)
    text(ax, cx + 0.62, top + S - 0.30, r"$\pi(v)=u(v)\,I(v)$", fs=FS - 1)
    for c, (img, vals, name) in enumerate([(V, (0.70, 0.95, 0.67), "$v$"), (VP, (0.95, 0.0, 0.0), "$v′$")]):
        x = cx + c * 0.68
        yb = top - 1.40
        thumb(ax, img, x + 0.08, yb + 0.86, 0.30, lw=0.6)
        if c == 0:
            frame(ax, x + 0.08, yb + 0.86, 0.30)
        text(ax, x + 0.23, yb + 1.25, name)
        for j, (val, lab) in enumerate(zip(vals, ["$u$", "$I$", r"$\pi$"])):
            bxx = x + 0.04 + j * 0.14
            ax.add_patch(Rectangle((bxx, yb + 0.14), 0.09, 0.60 * val, fc=["0.25", "0.6", ACC][j],
                                   hatch=[None, None, "////"][j], ec="k" if val else "none", lw=0.4, zorder=4))
            text(ax, bxx + 0.045, yb + 0.05, lab, fs=FS - 1)
        ax.plot([x, x + 0.46], [yb + 0.14, yb + 0.14], color="k", lw=0.4)
    text(ax, cx + 0.62, top - 1.55, "margin ($\\max\\,u$) picks $v′$\ninfluence ($\\max\\,\\pi$) picks $v$",
         fs=FS - 1)


def b_discount(ax, y0, dx=5.00):
    top = y0 + 1.72
    ax.plot([dx - 0.12, dx - 0.12], [y0 + 0.20, y0 + 2.25], color="0.75", lw=0.5)
    text(ax, dx, top + S + 0.12, r"after picking $v^\star\!=v$: discount", ha="left", fs=FS - 1, weight="bold")
    text(ax, dx, top + S - 0.10, r"$u(i)\leftarrow u(i)\,(1-w(i\!\to\! v^\star))$", ha="left", fs=FS - 1)
    for r, (key, qi, lst, pos) in enumerate(ROWS[:3]):
        y = top - (r + 0.75) * PITCH
        thumb(ax, qi, dx, y, S, lw=0.6)
        gauge(ax, dx + S + 0.05, y, S, U[key] * (1 - RHO_ILL ** pos), ghost=U[key])
        text(ax, dx + S + 0.17, y + S / 2, [r"$v^\star$ at pos 1: doubt cleared", r"pos 2: mostly resolved",
                                            r"pos 3: partly resolved"][r], ha="left", fs=FS - 1)
    y = top - 3.75 * PITCH
    thumb(ax, V, dx, y, S, lw=0.6)
    frame(ax, dx, y, S)
    label_badge(ax, dx + S, y + S)
    gauge(ax, dx + S + 0.05, y, S, 0.0, ghost=U["v"])
    text(ax, dx + S + 0.17, y + S / 2, r"$v^\star$ joins $\mathcal{L}$: $u(v^\star)=0$", ha="left", fs=FS - 1)
    text(ax, dx, y - 0.17, r"recompute $\pi$ $\rightarrow$ next pick ($q$ per round)", ha="left", fs=FS - 1)


def draw_b(b1):
    def f(ax, y0):
        b_header(ax, y0)
        b1(ax, y0 + 0.14)
        b_compare(ax, y0 + 0.14)
        b_discount(ax, y0 + 0.14)
    return f


def make(name, draw_b_, H_b=2.64):
    H_a = 2.34
    fig, ax = canvas(W, H_a + H_b)
    draw_b_(ax, 0.0)
    ax.plot([0.05, W - 0.05], [H_b + 0.02, H_b + 0.02], color="0.75", lw=0.5)
    draw_a(ax, H_b)
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png", dpi=220)


if __name__ == "__main__":
    make("fig1_lists", draw_b(b1_lists))
    make("fig1_graph", draw_b(b1_graph))
