"""
Batch active learning strategies (each selects q nodes from `cand` at once).

All take the same inputs and return an array with q indices.
None of them uses the label of an unlabeled node.

  margin               1 - margin (Pedronette & Latecki 2021, eq. 5)
  coreset              greedy k-center on the features (Sener & Savarese, ICLR 2018)
  margin_div           margin + rank diversity: skips candidates that are in the
                       top-k of (or have in their top-k) a node already chosen in the
                       SAME batch; if not enough remain, completes by margin
  rank_influence       uncertainty(v) x influence(v), influence(v) = sum of the
                       uncertainty of the nodes u that have v in their top-k, weight p^(pos-1)
                       ("a doubtful node that appears in the lists of many doubtful nodes")
  rank_influence_greedy  same, but greedy: after choosing v, the uncertainty of the
                       u that have v in their top-k is multiplied by (1 - weight) -- they become
                       "partially covered" -- and the influence is recomputed

Initial set (cold start, no model):
  rank_probcover       ProbCover (Yehuda et al., NeurIPS 2022) with the ball of radius
                       delta replaced by the top-k list: v covers {v} U top-k(v);
                       greedy, maximizes the newly covered nodes
"""

import numpy as np
import scipy.sparse as sp

import reliability as rel


def uncertainty(logp):
    return 1.0 - rel.margin_pl21(logp)


def uncertainty_prob(logp):
    """1 - (p1 - p2), margem classica em probabilidade (Scheffer et al. 2001), em [0, 1].
    CORRECAO (2026-09-30): a margem do PL21 e' normalizada pela soma de |log p| de TODAS
    as classes e fica comprimida em ~[0.96, 1]; no produto com a influencia, a incerteza
    virava constante e o rank_influence degenerava em grau de entrada (Spearman = 1.0).
    Pra ordenar sozinha (pick_margin) isso nao importa; no produto importa."""
    p = np.sort(np.exp(logp), axis=1)
    return 1.0 - (p[:, -1] - p[:, -2])


def _order(score, rng):
    """indices em ordem decrescente de score, empates por sorteio."""
    return np.lexsort((rng.random(len(score)), -score))


def reverse_matrix(rks, k, p):
    """R[u, v] = p^(pos-1) se v esta' na posicao pos (1..k) da lista de u (sem o proprio u)."""
    rks = np.asarray(rks)
    n = rks.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = rks[:, 1:k + 1].reshape(-1)
    vals = np.tile(p ** np.arange(k), n)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def pick_margin(logp, cand, q, rng, **_):
    return cand[_order(uncertainty(logp)[cand], rng)[:q]]


def pick_coreset(cand, q, labeled, feats, **_):
    x = feats

    def d2(idx):                                  # distancia^2 de cada candidato a x[idx]
        return ((x[cand] - x[idx]) ** 2).sum(1)

    dmin = np.full(len(cand), np.inf)
    for j in labeled:
        dmin = np.minimum(dmin, d2(j))
    out = []
    for _ in range(q):
        i = int(np.argmax(dmin))
        out.append(cand[i])
        dmin = np.minimum(dmin, d2(cand[i]))
        dmin[i] = -np.inf
    return np.asarray(out)


def pick_margin_div(logp, cand, q, rng, labeled, rks, k_div=10, **_):
    rks = np.asarray(rks)
    top = rks[:, 1:k_div + 1]
    n = rks.shape[0]
    blocked = np.zeros(n, dtype=bool)
    # rev[v] = nos que tem v no top-k (lista reversa), via CSR
    rev = sp.csr_matrix((np.ones(top.size), (top.reshape(-1), np.repeat(np.arange(n), k_div))), shape=(n, n))

    def block(v):
        blocked[top[v]] = True
        blocked[rev.indices[rev.indptr[v]:rev.indptr[v + 1]]] = True
        blocked[v] = True

    # so' o lote bloqueia: bloquear os vizinhos dos ja' rotulados (5% de n)
    # cobriria quase o dataset inteiro e o metodo viraria a margem pura
    order = cand[_order(uncertainty(logp)[cand], rng)]
    out = []
    for v in order:
        if len(out) == q:
            break
        if not blocked[v]:
            out.append(v)
            block(v)
    if len(out) < q:                                # completa pela margem
        rest = [v for v in order if v not in set(out)]
        out += rest[: q - len(out)]
    return np.asarray(out)


def pick_rank_influence(logp, cand, q, rng, labeled, R, unc_fn=uncertainty, **_):
    unc = unc_fn(logp)
    unc[labeled] = 0.0
    infl = R.T @ unc
    return cand[_order((unc * infl)[cand], rng)[:q]]


def pick_rank_influence_greedy(logp, cand, q, rng, labeled, R, unc_fn=uncertainty, **_):
    unc = unc_fn(logp)
    unc[labeled] = 0.0
    Rc = R.tocsc()
    avail = np.zeros(len(unc), dtype=bool)
    avail[cand] = True
    out = []
    noise = rng.random(len(unc)) * 1e-9
    for _ in range(q):
        score = unc * (R.T @ unc) + noise
        score[~avail] = -np.inf
        v = int(np.argmax(score))
        out.append(v)
        avail[v] = False
        col = Rc.getcol(v)
        unc[col.indices] *= (1.0 - col.data)       # quem tem v perto fica "coberto"
        unc[v] = 0.0
    return np.asarray(out)


def initial_rank_probcover(pool, m, rks, k_cover=10, rng=None):
    rks = np.asarray(rks)
    n = rks.shape[0]
    balls = rks[:, :k_cover + 1]                                   # inclui o proprio v
    covered = np.zeros(n, dtype=bool)
    avail = np.zeros(n, dtype=bool)
    avail[pool] = True
    noise = (rng or np.random.default_rng(0)).random(n) * 1e-6
    out = []
    for _ in range(m):
        gain = (~covered[balls]).sum(axis=1).astype(float)
        v = int(np.argmax(np.where(avail, gain + noise, -np.inf)))
        out.append(v)
        avail[v] = False
        covered[balls[v]] = True
    return np.sort(np.asarray(out))


PICKERS = {
    "margin": pick_margin,
    "coreset": pick_coreset,
    "margin_div": pick_margin_div,
    "rank_influence": pick_rank_influence,
    "rank_influence_greedy": pick_rank_influence_greedy,
    # versoes corrigidas (incerteza com escala cheia); as de cima ficam registradas
    "rank_influence_v2": lambda **kw: pick_rank_influence(unc_fn=uncertainty_prob, **kw),
    "rank_influence_greedy_v2": lambda **kw: pick_rank_influence_greedy(unc_fn=uncertainty_prob, **kw),
}


def pick_influence_variant(logp, cand, q, rng, labeled, R, self_doubt=True, nbr_doubt=True,
                           discount=True, ignore=None, **_):
    """
    Versao parametrizada da influencia gulosa, para a ablacao (exp 07). Com tudo True e
    ignore=None e' IDENTICA a pick_rank_influence_greedy(unc_fn=uncertainty_prob)
    (conferido em tests/test_influence.py).
      self_doubt=False  prioridade = influencia(v)                     (sem a duvida propria)
      nbr_doubt=False   influencia conta 1 por vizinho reverso nao rotulado, em vez da duvida dele
                        (so' "popularidade" ponderada pela posicao)
      discount=False    calcula a prioridade 1 vez e pega as q maiores (sem desconto guloso)
      ignore            nos cuja duvida NAO entra na influencia (ex.: o teste -> versao indutiva);
                        eles continuam sem poder ser escolhidos (cand ja' exclui o teste)
    """
    u = uncertainty_prob(logp)
    u[labeled] = 0.0
    nb = u.copy() if nbr_doubt else np.ones(len(u))
    nb[labeled] = 0.0
    if ignore is not None:
        nb[ignore] = 0.0
    own = u if self_doubt else np.ones(len(u))
    avail = np.zeros(len(u), dtype=bool)
    avail[cand] = True
    noise = rng.random(len(u)) * 1e-9
    if not discount:
        score = own * (R.T @ nb) + noise
        score[~avail] = -np.inf
        return np.argsort(-score, kind="stable")[:q]
    Rc = R.tocsc()
    own = own.copy()
    out = []
    for _ in range(q):
        score = own * (R.T @ nb) + noise
        score[~avail] = -np.inf
        v = int(np.argmax(score))
        out.append(v)
        avail[v] = False
        col = Rc.getcol(v)
        factor = 1.0 - col.data
        nb[col.indices] *= factor
        if self_doubt:
            own[col.indices] *= factor             # na versao completa, u e' um vetor so'
        nb[v] = 0.0
        own[v] = 0.0
    return np.asarray(out)
