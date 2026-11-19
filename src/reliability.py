"""
Confidence signals c_i for the predicted label of each node.

All return an array (n,) in which LARGER = more reliable. None uses the true
label: only the output of the GCN (logp, hidden), the graph and the ranked lists.
In this repository, active.py uses only margin_pl21 (to compute the uncertainty);
the other signals come from an earlier stage of the project and are kept for completeness.

Families:
  internal to the GCN
    maxprob       maximum softmax
    entropy       1 - H(p)/log C (Iscen et al., CVPR 2019)
    margin_pl21   margin between the 2 smallest |log p| normalized by their sum
                  (Pedronette & Latecki, IP&M 2021, eq. 5)
  graph neighborhood
    graph_agree   fraction of the graph neighbors with the same predicted class
                  (simplified version of the idea of TA-GST/NeST)
  ranked lists
    rank_agree    class agreement with the top-k of tau_i, with position
                  decay p^(d-1) (same descriptor as the graph)
    rbo_out       RBO(tau_i, list of i in the output space of the GCN,
                  sqrt(p) = Hellinger)
    rbo_hidden    RBO(tau_i, list of i in the hidden layer, cosine)
    xdesc_agree   rank_agree with the lists of OTHER descriptors (mean)
    rank_bc       "soft" version of rank_agree: instead of [pred_j == pred_i],
                  the Bhattacharyya coefficient BC(p_i, p_j) = sum_c sqrt(p_ic p_jc)
                  = 1 - H^2(p_i, p_j) (Hellinger). Measures whether the neighbor has the
                  same DISTRIBUTION, not only the same winning class.
Note: sqrt is monotone, so sqrt(maxprob) ranks like maxprob (same AUROC);
Hellinger only changes something when comparing TWO nodes.

Ranked lists: rks[i, 0] = i (the node itself); position 0 is always discarded
here and rks[:, 1:k+1] is used.
"""

import numpy as np
import torch


# ---------------------------------------------------------------- internos

def maxprob(logp):
    return np.exp(logp.max(axis=1))


def entropy_conf(logp):
    p = np.exp(logp)
    h = -(p * logp).sum(axis=1)
    return 1.0 - h / np.log(logp.shape[1])


def margin_pl21(logp):
    """
    Eq. 5 de Pedronette & Latecki (2021): S_i = |log p_i|, c = (m2 - m1) / sum(S_i),
    m1 e m2 as duas menores entradas de S_i. (O artigo define m2 como o menor
    valor ESTRITAMENTE maior que m1; com floats empate exato nao acontece.)
    """
    s = np.abs(logp)
    two = np.partition(s, 1, axis=1)[:, :2]
    return (two[:, 1] - two[:, 0]) / s.sum(axis=1)


# ---------------------------------------------------------------- vizinhanca

def graph_agree(edge_index, pred):
    """Fracao dos vizinhos j (aresta j -> i, quem manda mensagem pra i) com pred_j == pred_i."""
    ei = edge_index.cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    src, dst = ei
    keep = src != dst
    src, dst = src[keep], dst[keep]
    n = len(pred)
    same = np.bincount(dst, weights=(pred[src] == pred[dst]).astype(float), minlength=n)
    deg = np.bincount(dst, minlength=n)
    return np.divide(same, deg, out=np.zeros(n), where=deg > 0)


def _decay(k, p):
    return p ** np.arange(k)


def rank_agree(rks, pred, k=20, p=0.9):
    """sum_d p^(d-1) [pred(tau_i(d)) == pred_i] / sum_d p^(d-1), d = 1..k (sem o proprio i)."""
    nb = np.asarray(rks)[:, 1:k + 1]
    w = _decay(k, p)
    return ((pred[nb] == pred[:, None]) * w).sum(axis=1) / w.sum()


def rank_bc(rks, logp, k=20, p=0.9):
    """sum_d p^(d-1) BC(p_i, p_tau_i(d)) / sum_d p^(d-1), BC = sum_c sqrt(p_ic p_jc)."""
    nb = np.asarray(rks)[:, 1:k + 1]
    q = np.sqrt(np.exp(logp))                               # (n, C), linhas com norma 1
    bc = np.einsum("nc,nkc->nk", q, q[nb])
    w = _decay(k, p)
    return (bc * w).sum(axis=1) / w.sum()


def xdesc_agree(other_rks, pred, k=20, p=0.9):
    """rank_agree medio sobre as ranked lists de outros descritores."""
    return np.mean([rank_agree(r, pred, k, p) for r in other_rks], axis=0)


# ---------------------------------------------------------------- RBO

def knn_lists(z, k, metric="cosine", device=None, chunk=2048):
    """
    Top-k vizinhos de cada no' no espaco z (sem o proprio no'), (n, k).
    metric: "cosine" ou "hellinger" (z = probabilidades; usa sqrt(z) + produto interno,
    que ordena igual a distancia de Hellinger).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.as_tensor(z, dtype=torch.float32, device=device)
    if metric == "hellinger":
        z = z.clamp_min(0).sqrt()
    elif metric == "cosine":
        z = torch.nn.functional.normalize(z, dim=1)
    else:
        raise ValueError(metric)
    n = z.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, chunk):
        sim = z[s:s + chunk] @ z.T
        rows = torch.arange(s, min(s + chunk, n), device=device)
        sim[rows - s, rows] = -float("inf")
        out[s:s + chunk] = sim.topk(k, dim=1).indices.cpu().numpy()
    return out


def rbo(a, b, p=0.9):
    """
    RBO truncado em k (Webber et al., TOIS 2010), normalizado pra [0, 1]:
        sum_d p^(d-1) |a[:d] & b[:d]| / d  /  sum_d p^(d-1),  d = 1..k
    a, b: (n, k), listas sem repeticao. Vetorizado: o par (a[x], b[y]) com
    a[x] == b[y] entra na intersecao a partir da profundidade max(x, y) + 1.
    """
    a, b = np.asarray(a), np.asarray(b)
    n, k = a.shape
    eq = a[:, :, None] == b[:, None, :]                     # (n, k, k)
    depth = np.maximum(np.arange(k)[:, None], np.arange(k)[None, :])  # max(x, y)
    counts = np.zeros((n, k))
    rows, xs, ys = np.nonzero(eq)
    np.add.at(counts, (rows, depth[xs, ys]), 1)
    inter = np.cumsum(counts, axis=1)                       # |a[:d] & b[:d]|, d = 1..k
    w = _decay(k, p)
    return (inter / np.arange(1, k + 1) * w).sum(axis=1) / w.sum()


def rbo_to_gcn(rks, z, k=20, p=0.9, metric="cosine"):
    return rbo(np.asarray(rks)[:, 1:k + 1], knn_lists(z, k, metric), p)


# ---------------------------------------------------------------- tudo junto

def all_signals(logp, hidden, edge_index, rks, other_rks, k=20, p=0.9):
    pred = logp.argmax(axis=1)
    sig = {
        "maxprob": maxprob(logp),
        "entropy": entropy_conf(logp),
        "margin_pl21": margin_pl21(logp),
        "graph_agree": graph_agree(edge_index, pred),
        "rank_agree": rank_agree(rks, pred, k, p),
        "rank_bc": rank_bc(rks, logp, k, p),
        "rbo_out": rbo_to_gcn(rks, np.exp(logp), k, p, metric="hellinger"),
        "rbo_hidden": rbo_to_gcn(rks, hidden, k, p, metric="cosine"),
    }
    if other_rks:
        sig["xdesc_agree"] = xdesc_agree(other_rks, pred, k, p)
    # combinacoes simples com o softmax (a pergunta: o sinal de rank acrescenta algo?)
    for s in ("rank_agree", "rank_bc", "rbo_out", "xdesc_agree"):
        if s in sig:
            sig[f"maxprob*{s}"] = sig["maxprob"] * sig[s]
    return sig
