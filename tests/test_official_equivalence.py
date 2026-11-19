"""
Equivalence test of the baselines against the OFFICIAL CODE (avihu111/TypiClust).

Two questions:
  A. Do our PORTS (src/official_ports.py) select exactly the same images as the
     official code when run? (they should: it is the same logic)
  B. How far from the official code are the implementations used in exp 05
     (src/selection.select_typiclust and src/coverage.select_probcover, written from the papers)?
     (overlap of the picks and the objective of each method)

The official repository must be cloned; pass the path of deep-al:
  OFFICIAL_DEEPAL=/path/TypiClust/deep-al python3 tests/test_official_equivalence.py
The faiss library (used only for exact kNN in the official TypiClust) is replaced by an
equivalent exact search in numpy, without changing their code.
"""

import os
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
DEEPAL = os.environ.get("OFFICIAL_DEEPAL")
if not DEEPAL or not os.path.isdir(DEEPAL):
    sys.exit("defina OFFICIAL_DEEPAL=/caminho/para/TypiClust/deep-al")
sys.path.insert(0, DEEPAL)


# ---- faiss falso: IndexFlatL2 exato (distancia L2 AO QUADRADO, como o faiss) ----
class _IndexFlatL2:
    def __init__(self, d):
        self.x = None

    def add(self, x):
        self.x = np.asarray(x, dtype=np.float64)

    def search(self, q, k):
        q = np.asarray(q, dtype=np.float64)
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ self.x.T + (self.x ** 2).sum(1)[None, :]
        d2 = np.maximum(d2, 0)
        idx = np.argsort(d2, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(d2, idx, 1).astype(np.float32), idx


sys.modules["faiss"] = types.SimpleNamespace(IndexFlatL2=_IndexFlatL2, index_cpu_to_all_gpus=lambda i: i)

import data  # noqa: E402
import coverage as cv  # noqa: E402
import selection  # noqa: E402
import official_ports as op  # noqa: E402
from protocol import fixed_test_split  # noqa: E402
import pycls.al.typiclust as off_typ  # noqa: E402
import pycls.al.prob_cover as off_pc  # noqa: E402


def overlap(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(len(a), 1)


def pool_coverage(sel, pool, feats_n, delta):
    """Objetivo do ProbCover: fracao do pool a menos de delta de algum escolhido."""
    xs, ys, _ = op.radius_graph(feats_n, np.concatenate([np.asarray(sel), np.setdiff1d(pool, sel)]), delta)
    cov = np.zeros(len(pool), dtype=bool)
    cov[ys[xs < len(sel)]] = True
    return cov.mean()


def main():
    for ds in ["flowers", "corel5k"]:
        labels = data.load_labels(ds)
        feats = data.load_features(ds, "Resnet152")
        fn = op.l2_normalize(feats)
        pool, _ = fixed_test_split(labels, 0)
        uset = np.sort(pool)
        C = int(labels.max()) + 1
        for frac in (0.02, 0.05):
            m = int(round(frac * len(labels)))
            seed = 0
            cfg = {"DATASET": {"NAME": "x"}, "RNG_SEED": seed}

            # ---------------- TypiClust
            off_typ.ds_utils.load_features = lambda *a, **k: fn
            off_typ.kmeans = lambda f, num_clusters: op.typiclust_kmeans(f, num_clusters, seed)
            t = off_typ.TypiClust(cfg, np.array([], dtype=int), uset, m)
            official, _ = t.select_samples()
            port = op.typiclust_official(pool, m, feats, seed=seed)
            ours = np.asarray(selection.select_typiclust(pool, m, seed, feats, tag="eq"))
            selection.clear_cache()
            print(f"{ds} m={m} TypiClust: porte vs oficial = {overlap(port, official):.3f} "
                  f"(identico: {sorted(port) == sorted(official)}) | exp05 vs oficial = {overlap(ours, official):.3f} "
                  f"| classes cobertas: oficial {len(np.unique(labels[official]))}/{C}, exp05 {len(np.unique(labels[ours]))}/{C}")

            # ---------------- ProbCover (mesmo delta)
            delta = cv.probcover_delta(feats, C, seed=seed, tag=f"eq{ds}")
            off_pc.ds_utils.load_features = lambda *a, **k: fn
            p = off_pc.ProbCover(cfg, np.array([], dtype=int), uset, m, delta)
            official_bug, _ = p.select_samples()             # como esta' no repositorio
            p.graph_df = p.construct_graph(batch_size=len(uset))   # sem o bug do ultimo lote
            official, _ = p.select_samples()
            port = op.probcover_official(pool, m, feats, delta)
            ours, _ = cv.select_probcover(pool, m, feats, C, seed=seed, tag=f"eq{ds}")
            print(f"{ds} m={m} ProbCover (delta={delta:.3f}): porte vs oficial = {overlap(port, official):.3f} "
                  f"(identico: {sorted(port) == sorted(official)}) | oficial com bug do lote vs sem = "
                  f"{overlap(official_bug, official):.3f} | exp05 vs oficial = {overlap(ours, official):.3f}")
            print(f"    cobertura do pool: oficial {pool_coverage(official, pool, fn, delta):.3f}, "
                  f"exp05 {pool_coverage(ours, pool, fn, delta):.3f}")


if __name__ == "__main__":
    main()
