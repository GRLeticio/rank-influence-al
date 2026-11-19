"""Drawing primitives for Figure 1. Everything in inches, drawn at the final size."""
import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Polygon
from PIL import Image

IMG_DIR = os.path.join(os.environ.get("FLOWERS_IMAGES_DIR", "datasets/flowers/jpg"), "")   # imagens do Flowers-17 (image_0001.jpg ...)

FS = 8          # tamanho base (pt) -- ajustar para o tamanho do texto do PRL
FS_T = 9        # titulos de painel
ACC = "#D55E00"  # destaque (Okabe-Ito vermelhao): so' para a contribuicao / v
SEC = "#0072B2"  # secundaria (Okabe-Ito azul)
GRAY = "0.55"
LIGHT = "0.92"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

_cache = {}


def load(i, px=160):
    if i not in _cache:
        im = Image.open(IMG_DIR + f"image_{i:04d}.jpg").convert("RGB")
        w, h = im.size
        m = min(w, h)
        im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((px, px), Image.LANCZOS)
        _cache[i] = np.asarray(im)
    return _cache[i]


def canvas(W, H):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def thumb(ax, i, x, y, s, ec="k", lw=0.4, z=3, alpha=1.0):
    """miniatura com canto inferior esquerdo em (x, y)."""
    ax.imshow(load(i), extent=[x, x + s, y, y + s], zorder=z, interpolation="bilinear", alpha=alpha)
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=ec, lw=lw, zorder=z + 1))


def frame(ax, x, y, s, ec=ACC, lw=1.6, pad=0.018, z=6):
    ax.add_patch(Rectangle((x - pad, y - pad), s + 2 * pad, s + 2 * pad, fill=False, ec=ec, lw=lw, zorder=z))


def gauge(ax, x, y, h, frac, w=0.055, ghost=None, z=5):
    """barra vertical de duvida u (0..1). ghost = valor anterior (contorno tracejado)."""
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="k", lw=0.4, zorder=z))
    if ghost is not None and ghost > frac:
        ax.add_patch(Rectangle((x, y), w, h * ghost, fc="none", ec="0.35", lw=0.4, hatch="//////", zorder=z + 1))
    if frac > 0:
        ax.add_patch(Rectangle((x, y), w, h * frac, fc="0.25", ec="none", zorder=z + 2))


def label_badge(ax, x, y, r=0.045, z=8):
    """marca de 'rotulado': quadradinho preto com check, no canto superior direito."""
    ax.add_patch(Rectangle((x - r, y - r), 2 * r, 2 * r, fc="k", ec="white", lw=0.4, zorder=z))
    ax.plot([x - 0.55 * r, x - 0.1 * r, x + 0.6 * r], [y, y - 0.45 * r, y + 0.5 * r], color="white", lw=0.8,
            solid_capstyle="round", zorder=z + 1)


def box(ax, x, y, w, h, text=None, fc="white", ec="k", lw=0.6, ls="-", r=0.04, fs=FS, z=2, tc="k", weight="normal",
        ha="center", va="center", tx=None, ty=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))
    if text:
        ax.text(x + w / 2 if tx is None else tx, y + h / 2 if ty is None else ty, text, ha=ha, va=va, fontsize=fs,
                color=tc, weight=weight, zorder=z + 1, linespacing=1.15)


def arrow(ax, p, q, lw=0.7, color="k", ls="-", head=5, rad=0.0, z=4, shrinkA=0, shrinkB=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={head / 20:.2f},head_width={head / 40:.2f}",
                                 mutation_scale=10, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=shrinkA, shrinkB=shrinkB))


def text(ax, x, y, s, fs=FS, ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=fs, ha=ha, va=va, zorder=kw.pop("zorder", 7), linespacing=1.15, **kw)


def panel_title(ax, x, y, s):
    text(ax, x, y, s, fs=FS_T, ha="left", va="top", weight="bold")
