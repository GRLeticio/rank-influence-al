"""
Coverage-based selection on the FEATURES (baselines from the low-budget AL literature).

Implemented from the papers (not from the official code); differences are
stated here.

  probcover   Yehuda et al., "Active Learning Through a Covering Lens", NeurIPS 2022.
              L2-normalized features; graph of balls of radius delta; delta = largest
              value with purity >= ALPHA (0.95), purity estimated with k-means (K = number
              of classes) as pseudo-labels, as in the paper; greedy: picks the node that
              covers the most uncovered nodes; when no node covers anything new, picks at random.
              Approximation: a ball only sees the KNN_L nearest neighbors
              (with a purity delta of 0.95 the balls are much smaller than that).
  maxherding  Bae, Noh & Sutherland, "Generalized Coverage for More Robust Low-Budget
              Active Learning", ECCV 2024. Generalized coverage with a Gaussian kernel:
              maximizes sum_i max_{s in S} k(x_i, s); greedy. Kernel sparsified to the
              KNN_L neighbors. Bandwidth sigma = median distance to the 20th neighbor
              (our heuristic, fixed before running; the paper shows that the method is
              not very sensitive to it -- precisely its advantage over ProbCover).

Both accept `labeled` (labeled nodes count as covered), so they serve
both for the initial set and for later rounds.
"""

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

KNN_L = 200
ALPHA = 0.95
SIGMA_K = 20


def _knn_with_dist(feats, L=KNN_L, device=None, chunk=2048):
    """(idx, dist) dos L vizinhos mais proximos de cada no' (sem ele mesmo), features L2-normalizadas."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.nn.functional.normalize(torch.as_tensor(feats, dtype=torch.float32, device=device), dim=1)
    n = z.shape[0]
    L = min(L, n - 1)
    idx = np.empty((n, L), dtype=np.int64)
    dist = np.empty((n, L), dtype=np.float32)
    for s in range(0, n, chunk):
        sim = z[s:s + chunk] @ z.T
        rows = torch.arange(s, min(s + chunk, n), device=device)
        sim[rows - s, rows] = -float("inf")
        v, i = sim.topk(L, dim=1)
        idx[s:s + chunk] = i.cpu().numpy()
        dist[s:s + chunk] = torch.sqrt(torch.clamp(2 - 2 * v, min=0)).cpu().numpy()
    return idx, dist


_CACHE = {}


def _neighbors(feats, tag):
    if tag not in _CACHE:
        _CACHE.clear()
        _CACHE[tag] = _knn_with_dist(feats)
    return _CACHE[tag]


def probcover_delta(feats, n_classes, seed=0, tag=None):
    idx, dist = _neighbors(feats, tag)
    z = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    pseudo = KMeans(n_clusters=n_classes, n_init=1, random_state=seed).fit_predict(z)
    diff = pseudo[idx] != pseudo[:, None]                  # (n, L), vizinhos em ordem de distancia
    has = diff.any(axis=1)
    first = np.argmax(diff, axis=1)
    # raio ate' o 1o vizinho de outro pseudo-rotulo: bolas menores que isso sao puras
    first_bad = np.where(has, dist[np.arange(len(dist)), first], np.inf)
    # pureza(delta) = fracao de nos com bola pura = mean(first_bad >= delta);
    # maior delta com pureza >= ALPHA:
    return float(np.quantile(first_bad, 1 - ALPHA))


def select_probcover(pool, m, feats, n_classes, labeled=(), seed=0, tag=None, rng=None, ordered=False):
    """ordered=True devolve na ORDEM de escolha (a de tamanho m e' prefixo da de tamanho M)."""
    rng = rng or np.random.default_rng(seed)
    idx, dist = _neighbors(feats, tag)
    delta = probcover_delta(feats, n_classes, seed, tag)
    n = len(feats)
    inball = dist < delta                                    # (n, L): j esta' na bola de i
    covered = np.zeros(n, dtype=bool)
    for s in labeled:
        covered[s] = True
        covered[idx[s][inball[s]]] = True
    avail = np.zeros(n, dtype=bool)
    avail[pool] = True
    avail[list(labeled)] = False
    out = []
    for _ in range(m):
        gain = ((~covered[idx]) & inball).sum(axis=1) + (~covered)  # vizinhos novos + ele mesmo
        gain = np.where(avail, gain, -1)
        if gain.max() <= 0:
            v = int(rng.choice(np.flatnonzero(avail)))
        else:
            best = np.flatnonzero(gain == gain.max())
            v = int(rng.choice(best))
        out.append(v)
        avail[v] = False
        covered[v] = True
        covered[idx[v][inball[v]]] = True
    out = np.asarray(out)
    return (out if ordered else np.sort(out)), delta


def select_maxherding(pool, m, feats, labeled=(), tag=None, rng=None, ordered=False):
    rng = rng or np.random.default_rng(0)
    idx, dist = _neighbors(feats, tag)
    n = len(feats)
    sigma = float(np.median(dist[:, SIGMA_K - 1]))
    kv = np.exp(-dist.astype(np.float64) ** 2 / (2 * sigma ** 2))  # k(x_i, x_idx[i,j])
    K = sp.csr_matrix((kv.reshape(-1), (np.repeat(np.arange(n), idx.shape[1]), idx.reshape(-1))), shape=(n, n))
    K = K.maximum(K.T) + sp.identity(n, format="csr")     # simetrico; k(x, x) = 1
    coo = K.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    Kc = K.tocsc()

    def column(v):
        a, b = Kc.indptr[v], Kc.indptr[v + 1]
        return Kc.indices[a:b], Kc.data[a:b]

    cur = np.zeros(n)                                         # max_{s in S} k(x_i, s)
    for s_ in labeled:
        r, d = column(s_)
        np.maximum.at(cur, r, d)
    avail = np.zeros(n, dtype=bool)
    avail[pool] = True
    avail[list(labeled)] = False
    out = []
    noise = rng.random(n) * 1e-12
    for _ in range(m):
        gain = np.bincount(cols, weights=np.maximum(vals - cur[rows], 0), minlength=n) + noise
        gain[~avail] = -np.inf
        v = int(np.argmax(gain))
        out.append(v)
        avail[v] = False
        r, d = column(v)
        np.maximum.at(cur, r, d)
    out = np.asarray(out)
    return (out if ordered else np.sort(out)), sigma
