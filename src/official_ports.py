"""
Baselines ported from the OFFICIAL CODE of the authors (not from the text of the papers).

Each function states the file it came from and what was changed. Only what is
needed to fit our protocol changes (precomputed features, rank-based graph
instead of Cora, fixed budget); the selection logic is theirs.

  typiclust_official   avihu111/TypiClust  deep-al/pycls/al/typiclust.py      (ICML 2022)
  probcover_official   avihu111/TypiClust  deep-al/pycls/al/prob_cover.py     (NeurIPS 2022)
  DCoMState            avihu111/TypiClust  deep-al/pycls/al/DCoM.py + tools/train_al.py (2024)
  age_query            vwz/AGE             algcn/train_entropy_density_graphcentral_ts.py (2017)
  rim_select           zwt233/RIM          examples/RIM.py                    (NeurIPS 2021)
  grain_select         zwt233/Grain        examples/GRAIN(ball-D).py          (VLDB 2021)
  info_density_query   Settles & Craven, EMNLP 2008 (no code; formula of the paper)

Difference common to all: the "features" are those of the descriptor (ResNet/ViT/Swin),
not the self-supervised ones (SimCLR/DINO) of the TypiClust/ProbCover/DCoM papers;
the graph is the reciprocal kNN graph (k=40) of the ranked lists (the same as the GCN),
not Cora/PubMed.
"""

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from sklearn.cluster import KMeans, MiniBatchKMeans


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _argmax(score, rng=None):
    """argmax com desempate. rng=None: menor indice (np.argmax, comportamento do codigo oficial).
    Com rng: sorteio entre os empatados. Nos nossos datasets as imagens estao ordenadas por
    classe (Flowers: 80 seguidas da mesma classe), entao "menor indice" nao e' neutro: empurra
    as escolhas empatadas sempre pras primeiras classes. Conferido no Corel5k/ResNet: ProbCover
    oficial tem empate no maximo em 228 de 250 escolhas, RIM em 203 de 250."""
    score = np.asarray(score, dtype=float)
    if rng is None:
        return int(np.argmax(score))
    best = np.flatnonzero(score == score.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def l2_normalize(feats):
    # load_features(..., normalized=True) do repositorio TypiClust
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


# ============================================================ TypiClust (oficial)

TYPI_MIN_CLUSTER_SIZE = 5
TYPI_MAX_NUM_CLUSTERS = 500
TYPI_K_NN = 20


def _typicality(rel_feats, k):
    """1 / (media das k menores distancias L2 AO QUADRADO + 1e-5): faiss.IndexFlatL2 devolve
    distancia ao quadrado, e o codigo oficial usa direto."""
    x = torch.as_tensor(rel_feats, dtype=torch.float32, device=_device())
    d2 = torch.cdist(x, x).pow(2)
    d2.fill_diagonal_(float("inf"))
    knn = d2.topk(k, dim=1, largest=False).values
    return (1.0 / (knn.mean(1) + 1e-5)).cpu().numpy()


def typiclust_kmeans(X, num_clusters, seed):
    """kmeans() de typiclust.py, com os mesmos parametros (defaults do sklearn) + random_state."""
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, random_state=seed)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, random_state=seed)
    km.fit_predict(X)
    return km.labels_


def typiclust_official(pool, m, feats, labeled=(), seed=0):
    """
    Porte de TypiClust.select_samples (k-means em todas as features, ate' 500 clusters,
    descarta clusters com <= 5, ordena por (ja' rotulados, -tamanho), tipicidade com distancia
    ao quadrado). Mudancas: k-means com random_state=seed (o
    oficial nao fixa semente); kNN exato com torch em vez de faiss (mesmo resultado);
    cluster com 1 elemento e' escolhido direto (o oficial quebraria com k=0).
    """
    labeled = np.asarray(labeled, dtype=int)
    unl = np.setdiff1d(pool, labeled)
    relevant = np.concatenate([labeled, unl]).astype(int)
    allX = l2_normalize(feats)
    X = allX[relevant]
    num_clusters = min(len(labeled) + m, TYPI_MAX_NUM_CLUSTERS)
    # o oficial agrupa TODAS as features do dataset (inclusive teste, sem rotulo) e depois
    # pega so' as relevantes
    labels = typiclust_kmeans(allX, num_clusters, seed)[relevant].copy()
    existing = np.arange(len(labeled))
    ids, sizes = np.unique(labels, return_counts=True)
    counts = np.bincount(labels[existing], minlength=ids.max() + 1)[ids] if len(existing) else np.zeros(len(ids), int)
    order = sorted([(int(c), -int(s), int(i)) for i, s, c in zip(ids, sizes, counts) if s > TYPI_MIN_CLUSTER_SIZE])
    order = [i for _, _, i in order]
    labels[existing] = -1
    selected = []
    for i in range(m):
        cluster = order[i % len(order)]
        idx = np.flatnonzero(labels == cluster)
        if len(idx) == 0:                      # cluster esgotado: vai pro proximo com gente
            alive = [c for c in order if (labels == c).any()]
            cluster = alive[i % len(alive)]
            idx = np.flatnonzero(labels == cluster)
        if len(idx) == 1:
            pick = idx[0]
        else:
            pick = idx[_typicality(X[idx], min(TYPI_K_NN, len(idx) // 2)).argmax()]
        selected.append(pick)
        labels[pick] = -1
    return relevant[np.asarray(selected)]


# ============================================================ ProbCover (oficial)

def radius_graph(feats_n, nodes, delta, chunk=1024):
    """Arestas x->y (indices locais em `nodes`) com ||x - y|| < delta, distancia exata (cdist em blocos)."""
    x = torch.as_tensor(feats_n[nodes], dtype=torch.float32, device=_device())
    xs, ys, ds = [], [], []
    for s in range(0, len(nodes), chunk):
        d = torch.cdist(x[s:s + chunk], x)
        mask = d < delta
        a, b = mask.nonzero(as_tuple=True)
        xs.append((a + s).cpu().numpy())
        ys.append(b.cpu().numpy())
        ds.append(d[mask].cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)


def probcover_official(pool, m, feats, delta, labeled=(), rng=None):
    """
    Porte de ProbCover.select_samples: grafo x->y se L2 < delta (features normalizadas),
    SO' entre os nos relevantes (rotulados + pool, sem o teste); guloso pelo maior grau de
    saida para nos ainda descobertos; empate = menor indice (np.argmax, como no oficial).
    delta: o oficial recebe como hiperparametro; aqui vem da regra de pureza do artigo
    (coverage.probcover_delta), a mesma usada no nosso ProbCover.
    """
    labeled = np.asarray(labeled, dtype=int)
    unl = np.setdiff1d(pool, labeled)
    relevant = np.concatenate([labeled, unl]).astype(int)
    N = len(relevant)
    xs, ys, _ = radius_graph(l2_normalize(feats), relevant, delta)
    G = sp.csr_matrix((np.ones(len(xs), dtype=np.int32), (xs, ys)), shape=(N, N))
    covered = np.zeros(N, dtype=bool)
    if len(labeled):
        covered[G[np.arange(len(labeled))].indices] = True
    selected = []
    for _ in range(m):
        deg = G @ (~covered).astype(np.int32)
        deg[selected] = -1                       # o oficial nao re-escolhe (grau 0 depois de coberto)
        deg[:len(labeled)] = -1
        cur = _argmax(deg, rng)
        covered[G[cur].indices] = True
        selected.append(cur)
    return relevant[np.asarray(selected)]


# ============================================================ DCoM (oficial)

DCOM_MAX_DELTA = 1.1
DCOM_DELTA_RESOLUTION = 0.05
DCOM_K_LOGISTIC = 50
DCOM_A_LOGISTIC = 0.8


class DCoMState:
    """
    Porte de DCoM (DCoM.py) + o ciclo de atualizacao dos deltas (tools/train_al.py).
    Cada rotulado tem o seu delta. Ciclo por rodada:
      select(): competence = logistica(cobertura atual); escolhe guloso o maximo de
                competence * margem_norm + (1 - competence) * densidade_norm
      update_deltas(): depois de treinar, busca binaria do delta de cada novo centro
                pela pureza dos pseudo-rotulos (limiar 0.2 * cobertura + 0.4)
    Inicial (sem rotulados): o oficial roda o mesmo select com um modelo nao treinado
    e competence ~ 0 -> cobertura pura com INITIAL_DELTA; aqui INITIAL_DELTA = delta de
    pureza do ProbCover (o oficial usa 0.6 pras features SimCLR deles).
    Mudanca: as distancias sao calculadas sob demanda (centro -> todos), sem montar o
    grafo inteiro de raio MAX_DELTA (mesmo resultado, cabe na memoria no CUB).
    """

    def __init__(self, pool, feats, initial_delta, rng=None):
        self.pool = np.asarray(pool, dtype=int)
        self.Xn = l2_normalize(feats)
        self.X = torch.as_tensor(self.Xn[self.pool], dtype=torch.float32, device=_device())
        self.pos = {int(v): i for i, v in enumerate(self.pool)}
        self.labeled = []                 # indices globais, em ordem de entrada
        self.deltas = []                  # delta de cada rotulado
        self.initial_delta = float(initial_delta)
        self.rng = rng

    def _dist_from(self, nodes_global):
        idx = torch.as_tensor([self.pos[int(v)] for v in nodes_global], device=self.X.device)
        return torch.cdist(self.X[idx], self.X)                      # (len, |pool|)

    def _covered(self, cap=None):
        """Pontos dentro da bola de algum rotulado (raio = delta proprio). cap: o grafo de densidade
        do original so' tem arestas com d < delta_avg, entao la' a bola fica limitada a cap."""
        cov = torch.zeros(len(self.pool), dtype=torch.bool, device=self.X.device)
        for s in range(0, len(self.labeled), 512):
            d = self._dist_from(self.labeled[s:s + 512])
            dl = torch.as_tensor(self.deltas[s:s + 512], device=d.device)[:, None]
            if cap is not None:
                dl = torch.clamp(dl, max=cap)
            cov |= (d < dl).any(0)
        return cov

    @staticmethod
    def _minmax(v):
        v = np.asarray(v, dtype=np.float64)
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

    def select(self, m, logp=None):
        delta_avg = float(np.mean(self.deltas)) if self.deltas else self.initial_delta
        zeros = torch.zeros(len(self.pool), dtype=torch.bool, device=self.X.device)
        coverage = float(self._covered().float().mean()) if self.labeled else 0.0   # competencia: raio proprio
        covered = self._covered(cap=delta_avg) if self.labeled else zeros            # grafo de densidade
        k, a = DCOM_K_LOGISTIC, DCOM_A_LOGISTIC
        p = 1 + np.exp(-k * (1 - a))
        competence = round(p / (1 + np.exp(-k * (coverage - a))), 3) if logp is not None else 0.0
        if logp is not None:
            prob = np.sort(np.exp(logp[self.pool]), axis=1)
            margin = self._minmax(-(prob[:, -1] - prob[:, -2]))      # "1 - margem", normalizada
        else:
            margin = np.zeros(len(self.pool))
        lab_mask = np.zeros(len(self.pool), dtype=bool)
        lab_mask[[self.pos[v] for v in self.labeled]] = True
        margin[lab_mask] = 0
        # grafo de raio delta_avg sem arestas saindo dos rotulados nem chegando nos cobertos
        xs, ys, _ = radius_graph(self.Xn, self.pool, delta_avg)
        keep = ~lab_mask[xs] & ~covered.cpu().numpy()[ys]
        G = sp.csr_matrix((np.ones(keep.sum(), dtype=np.int32), (xs[keep], ys[keep])),
                          shape=(len(self.pool),) * 2)
        alive = np.ones(len(self.pool), dtype=bool)
        alive[covered.cpu().numpy()] = False
        selected = []
        for _ in range(m):
            dens = G @ alive.astype(np.int32)
            score = competence * self._minmax(margin) + (1 - competence) * self._minmax(dens)
            score[lab_mask] = -np.inf
            score[selected] = -np.inf
            if self.rng is None:
                # desempate do original: argsort(...)[::-1][0] na ordem (rotulados, depois o resto)
                order = np.concatenate([[self.pos[v] for v in self.labeled],
                                        np.flatnonzero(~lab_mask)]).astype(int)
                sc = np.where(np.isfinite(score), score, -1e18)[order]
                cur = int(order[np.argsort(sc)[::-1][0]])
            else:
                cur = _argmax(np.where(np.isfinite(score), score, -1e18), self.rng)
            alive[G[cur].indices] = False
            margin[cur] = 0
            selected.append(cur)
        new = [int(self.pool[i]) for i in selected]
        self.labeled += new
        self.deltas += [delta_avg] * len(new)       # train_al.py: novos centros comecam com delta_avg
        return np.asarray(new), competence, coverage

    def update_deltas(self, n_new, labels, pseudo):
        """Busca binaria do delta dos n_new ultimos centros (new_centroids_deltas). labels: rotulo
        VERDADEIRO dos rotulados (legitimo); pseudo: classe prevista pela GCN pros outros."""
        coverage = float(self._covered().float().mean())
        thr = 0.2 * coverage + 0.4
        pool_pseudo = pseudo[self.pool]
        for j in range(len(self.labeled) - n_new, len(self.labeled)):
            c = self.labeled[j]
            d = self._dist_from([c])[0].cpu().numpy()
            within = d < DCOM_MAX_DELTA
            cent_label = labels[c]

            def purity(delta):
                nb = np.flatnonzero(within & (d < delta))
                return (pool_pseudo[nb] == cent_label).mean() if len(nb) else 0.0

            lo, hi = 0.0, DCOM_MAX_DELTA
            mid = (lo + hi) / 2
            last_p, last_d = 0.0, mid
            while abs(lo - hi) > DCOM_DELTA_RESOLUTION:
                cur = purity(mid)
                if last_d < mid and last_p == thr and cur < thr:
                    mid = last_d
                    break
                if cur < thr:
                    hi = mid
                else:
                    lo = mid
                last_p, last_d = cur, mid
                mid = (lo + hi) / 2
            self.deltas[j] = round(mid, 2)


# ============================================================ AGE (oficial)

AGE_BASEF = 0.995          # valor do AGE pra Cora e PubMed (0.9 no Citeseer)


def pagerank_normalized(edge_index, n, alpha=0.85, iters=100):
    """nx.pagerank no grafo nao dirigido, normalizado pra [0, 1] (centralissimo do AGE)."""
    ei = edge_index.cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    keep = ei[0] != ei[1]
    A = sp.csr_matrix((np.ones(keep.sum()), (ei[0][keep], ei[1][keep])), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    P = sp.diags(1 / np.maximum(deg, 1)) @ A
    r = np.full(n, 1 / n)
    for _ in range(iters):
        r = alpha * (P.T @ r) + alpha * r[deg == 0].sum() / n + (1 - alpha) / n
    return (r - r.min()) / (r.max() - r.min())


def _perc(v):
    """perc(input, k) do AGE: fracao dos valores ESTRITAMENTE menores que input[k]."""
    return (rankdata(v, method="min") - 1) / len(v)


def _percd(v):
    """percd: fracao dos valores estritamente maiores."""
    return (len(v) - rankdata(v, method="max")) / len(v)


def age_query(logp, cand, q, n_classes, centrality, epoch0, rng, seed=0):
    """
    Porte do laco de selecao do AGE: uma escolha por "epoca" com
    gamma ~ Beta(1, 1.005 - basef^epoca), alpha = beta = (1 - gamma) / 2,
    peso = alpha * perc(entropia) + beta * percd(dist. ao centro k-means das saidas) +
           gamma * perc(PageRank).
    Mudanca: o AGE escolhe 1 no' por epoca de treino; aqui o modelo e' fixo dentro da
    rodada, entao cada escolha da rodada usa a sua propria "epoca" (epoch0 + j).
    """
    p = np.exp(logp)
    ent = -(p * logp).sum(1)
    km = KMeans(n_clusters=n_classes, n_init=1, random_state=seed).fit(p)
    ed = _min_center_dist(p, km.cluster_centers_)
    entr_p, ed_p, cen_p = _perc(ent), _percd(ed), _perc(centrality)
    avail = np.zeros(len(p), dtype=bool)
    avail[cand] = True
    out = []
    for j in range(q):
        gamma = rng.beta(1, 1.005 - AGE_BASEF ** (epoch0 + j))
        alpha = beta = (1 - gamma) / 2
        w = alpha * entr_p + beta * ed_p + gamma * cen_p
        w[~avail] = -100
        s = int(np.argmax(w))
        out.append(s)
        avail[s] = False
    return np.asarray(out)


def _min_center_dist(p, centers, chunk=512):
    out = np.empty(len(p))
    for s in range(0, len(p), chunk):
        out[s:s + chunk] = ((p[s:s + chunk, None, :] - centers[None]) ** 2).sum(-1).min(1)
    return out


# ============================================================ RIM (oficial, oraculo sem ruido)

def _adj(edge_index, n):
    ei = edge_index.cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    keep = ei[0] != ei[1]
    A = sp.csr_matrix((np.ones(keep.sum()), (ei[0][keep], ei[1][keep])), shape=(n, n))
    return ((A + A.T) > 0).astype(np.float64)


def rim_select(pool, m, edge_index, n, th, rng, tie_rng=None, batch_size=5):
    """
    Porte de RIM.py com oracle_acc = 1 (nossos rotulos sao exatos): a confiabilidade vira 1 e o
    criterio e' cobertura gulosa -- v ativa j se (D^-1 (A+I))^2 [v, j] > th; escolhe quem ativa
    mais nos (todos os nos contam, como no oficial).
    Fiel ao codigo, inclusive numa peculiaridade dele: a cada `batch_size` escolhas o vetor de
    "ainda nao ativados" e' recalculado do zero (update_reliability) e, em seguida, a contribuicao
    do ultimo escolhido e' subtraida DE NOVO -- alguns valores ficam negativos e entram nas contas
    seguintes. Conferido contra o RIM.py original (tests/test_official_equivalence2.py).
    Mudancas: o oficial para quando ninguem ativa mais nada; aqui o orcamento e' completado por
    sorteio. Desempate: tie_rng=None = menor indice (oficial); com tie_rng, sorteio.
    th: o oficial usa 0.05 no Cora (grau medio ~4); o artigo manda ajustar por dataset.
    """
    A = _adj(edge_index, n) + sp.identity(n)
    Ar = sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
    A2 = (Ar @ Ar).tocsr()
    act = (A2 > th).astype(np.float64).tocsr()
    activated = np.ones(n)
    avail = np.zeros(n, dtype=bool)
    avail[pool] = True
    out = []
    stopped = False
    for count in range(1, m + 1):
        if not stopped:
            gain = act @ activated
            gain[~avail] = -np.inf
            v = _argmax(gain, tie_rng)
            if gain[v] <= 0:                       # o oficial para aqui
                stopped = True
        if stopped:
            v = int(rng.choice(np.flatnonzero(avail)))
            out.append(v)
            avail[v] = False
            continue
        row = act[v]
        act_vec = np.zeros(n)
        act_vec[row.indices] = activated[row.indices]
        out.append(v)
        avail[v] = False
        if count % batch_size == 0:
            ever = np.zeros(n)
            for u in out:
                ever[act[u].indices] = 1
            activated = 1.0 - ever
        activated = activated - act_vec
    return np.asarray(out)


def rim_threshold_median(edge_index, n):
    """th alternativo, fixado antes de rodar: mediana das entradas nao nulas de (D^-1(A+I))^2."""
    A = _adj(edge_index, n) + sp.identity(n)
    Ar = sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
    return float(np.median((Ar @ Ar).data))


# ============================================================ Grain (ball-D, oficial)

GRAIN_RADIUS = 0.05


def grain_select(pool, m, feats, edge_index, n, rng, radius=GRAIN_RADIUS, chunk=1024, tie_rng=None):
    """
    Porte de GRAIN(ball-D).py: features propagadas A_hat^2 X (A_hat = D^-1/2 (A+I) D^-1/2);
    distancias normalizadas pra [0, 1] pelo min/max global; bola(i) = {j: dist <= radius};
    v cobre a uniao das bolas dos seus vizinhos de 2 passos; guloso pela maior cobertura.
    Mudanca: completa o orcamento por sorteio se tudo ja' estiver coberto (o oficial para).
    """
    A = _adj(edge_index, n) + sp.identity(n)
    dinv = 1 / np.sqrt(np.asarray(A.sum(1)).ravel())
    Ah = sp.diags(dinv) @ A @ sp.diags(dinv)
    A2 = (Ah @ Ah).tocsr()
    Xp = A2 @ feats.astype(np.float64)
    x = torch.as_tensor(Xp, dtype=torch.float32, device=_device())
    # min/max global das distancias (o oficial normaliza a matriz inteira)
    dmin, dmax = np.inf, 0.0
    for s in range(0, n, chunk):
        d = torch.cdist(x[s:s + chunk], x)
        dmax = max(dmax, float(d.max()))
        dmin = min(dmin, float(d.min()))
    thr = dmin + radius * (dmax - dmin)
    rows, cols = [], []
    for s in range(0, n, chunk):
        a, b = (torch.cdist(x[s:s + chunk], x) <= thr).nonzero(as_tuple=True)
        rows.append((a + s).cpu().numpy())
        cols.append(b.cpu().numpy())
    B = sp.csr_matrix((np.ones(sum(len(r) for r in rows), dtype=np.float32),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    N2 = (A2 > 0).astype(np.float32)
    cover = ((N2 @ B) > 0).tocsr()                   # cover[v, i]: v cobre i
    uncovered = np.ones(n, dtype=bool)
    avail = np.zeros(n, dtype=bool)
    avail[pool] = True
    out = []
    for _ in range(m):
        gain = cover @ uncovered.astype(np.int32)
        gain[~avail] = -1
        v = _argmax(gain, tie_rng)
        if gain[v] <= 0:
            v = int(rng.choice(np.flatnonzero(avail)))
        out.append(v)
        avail[v] = False
        uncovered[cover[v].indices] = False
    return np.asarray(out)


# ============================================================ Information density

def info_density_query(logp, cand, q, feats, unlabeled, beta=1.0):
    """
    Settles & Craven (2008): incerteza(x) * (similaridade media de x com os nao rotulados)^beta.
    Incerteza = 1 - (p1 - p2) (a mesma do nosso metodo), similaridade = cosseno nas features.
    Com features normalizadas, a media do cosseno com U e' o produto interno com a media de U.
    """
    z = l2_normalize(feats)
    dens = z @ z[unlabeled].mean(0)
    p = np.sort(np.exp(logp), axis=1)
    unc = 1 - (p[:, -1] - p[:, -2])
    score = unc * np.clip(dens, 0, None) ** beta
    return cand[np.argsort(-score[cand], kind="stable")[:q]]


# ============================================================ MaxHerding (fiel ao artigo)

def maxherding_paper(pool, m, feats, labeled=(), sigma=1.0, rng=None, chunk=4096):
    """
    MaxHerding como no artigo (Bae, Noh & Sutherland, ECCV 2024; sem codigo publico -- o artigo
    diz que a implementacao parte do repositorio do TypiClust):
      cobertura generalizada C(L) = media sobre o pool de max_{x' em L} k(x, x'),
      k gaussiano exp(-||x - x'||^2 / (2 sigma^2)), sigma = 1 ("lengthscale fixed to 1"),
      features L2-normalizadas, kernel COMPLETO (sem cortar vizinhos), so' sobre o pool.
      Guloso: escolhe o x que mais aumenta C (Eq. 5 / Algoritmo 1).
    Diferente do coverage.select_maxherding do exp 05 (sigma = mediana da distancia ao 20o
    vizinho, kernel cortado em 200 vizinhos, todas as imagens): aquela fica como "nossa versao".
    """
    pool = np.asarray(pool, dtype=int)
    dev = _device()
    z = torch.as_tensor(l2_normalize(feats)[pool], dtype=torch.float32, device=dev)
    N = len(pool)
    K = torch.empty((N, N), dtype=torch.float32, device=dev)
    for s in range(0, N, chunk):
        K[s:s + chunk] = torch.exp(-torch.cdist(z[s:s + chunk], z).pow(2) / (2 * sigma ** 2))
    pos = {int(v): i for i, v in enumerate(pool)}
    cur = torch.zeros(N, device=dev)
    for v in labeled:
        cur = torch.maximum(cur, K[:, pos[int(v)]])
    avail = torch.ones(N, dtype=torch.bool, device=dev)
    for v in labeled:
        avail[pos[int(v)]] = False
    out = []
    for _ in range(m):
        gain = torch.zeros(N, device=dev)
        for s in range(0, N, chunk):                 # gain[c] = sum_n max(0, K[n, c] - cur[n])
            gain += torch.clamp(K[s:s + chunk] - cur[s:s + chunk, None], min=0).sum(0)
        g = gain.cpu().numpy().astype(np.float64)
        g[~avail.cpu().numpy()] = -np.inf
        c = _argmax(g, rng)
        out.append(int(pool[c]))
        avail[c] = False
        cur = torch.maximum(cur, K[:, c])
    del K
    return np.asarray(out)


# ============================================================ CoreSet (fiel ao codigo oficial)

def coreset_greedy(cand, q, labeled, emb):
    """greedy k-center de CoreSetMIPSampling (avihu111/TypiClust Sampling.py): distancia
    euclidiana AO QUADRADO; emb = REPRESENTACAO DO MODELO (o oficial usa a penultima camada
    do classificador treinado), aqui a camada escondida da GCN."""
    dev = _device()
    E = torch.as_tensor(emb, dtype=torch.float32, device=dev)
    C = E[torch.as_tensor(cand, device=dev)]
    dmin = torch.full((len(cand),), float("inf"), device=dev)
    for s in range(0, len(labeled), 1024):
        L = E[torch.as_tensor(np.asarray(labeled)[s:s + 1024], device=dev)]
        dmin = torch.minimum(dmin, torch.cdist(C, L).pow(2).min(1).values)
    out = []
    for _ in range(q):
        i = int(torch.argmax(dmin))
        out.append(int(cand[i]))
        dmin = torch.minimum(dmin, (C - C[i]).pow(2).sum(1))
        dmin[i] = -float("inf")
    return np.asarray(out)
