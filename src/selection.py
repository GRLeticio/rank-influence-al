"""
Strategies to choose WHICH nodes to label (budget m), always within the
pool (never in the fixed test set).

All return an ordered list of indices. The deterministic ones and random
selection are "nested": the selection with budget m is a prefix of the
selection with a larger budget, so it can be computed once with m_max and
sliced. stratified_oracle and kmeans do not have this property.

No strategy uses labels, except stratified_oracle (an oracle reference,
NOT a fair baseline).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def select_random(pool, m, seed):
    rng = np.random.default_rng(seed)
    return rng.permutation(pool)[:m].tolist()


def select_stratified_oracle(pool, m, seed, labels):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = {c: rng.permutation(pool[labels[pool] == c]).tolist() for c in classes}
    chosen = []
    # round-robin entre classes em ordem aleatoria: m//C por classe, resto espalhado
    while len(chosen) < m:
        for c in rng.permutation(classes):
            if per_class[c] and len(chosen) < m:
                chosen.append(per_class[c].pop())
    return chosen


_KM_CACHE = {}


def kmeans_fit(X, m, seed, tag):
    """
    KMeans(m) com cache: kmeans, typiclust e rank_typiclust usam o MESMO
    agrupamento (mesmo X, m, seed), entao a unica diferenca entre eles e' o
    criterio de escolha dentro do cluster. tag identifica X (ex.: "raw/split0").
    """
    key = (tag, m, seed)
    if key not in _KM_CACHE:
        km = KMeans(n_clusters=m, n_init=3, random_state=seed).fit(X)
        _KM_CACHE[key] = (km.labels_, km.cluster_centers_)
    return _KM_CACHE[key]


def clear_cache():
    _KM_CACHE.clear()


def select_kmeans(pool, m, seed, features, tag="raw"):
    # medoide = ponto do pool mais proximo de cada centroide (sem repetir)
    X = features[pool]
    _, centers = kmeans_fit(X, m, seed, tag)
    d = euclidean_distances(X, centers)
    chosen, taken = [], set()
    for c in range(m):
        for idx in np.argsort(d[:, c]):
            if idx not in taken:
                taken.add(idx)
                chosen.append(int(pool[idx]))
                break
    return chosen


def _one_per_cluster(pool, cluster_of, score):
    """Em cada cluster, o no' do pool com maior score."""
    chosen = []
    for c in np.unique(cluster_of):
        members = np.flatnonzero(cluster_of == c)
        chosen.append(int(pool[members[np.argmax(score[members])]]))
    return chosen


def select_typiclust(pool, m, seed, features, tag="raw", k_typ=20):
    """
    TypiClust (Hacohen et al., ICML 2022), versao cold start sem rotulos
    iniciais: m clusters, 1 escolha por cluster, a mais TIPICA =
    1 / distancia media aos k_typ vizinhos dentro do proprio cluster.
    (O original ignora clusters muito pequenos; aqui todo cluster recebe 1,
    pra fechar o orcamento m.)
    """
    X = features[pool]
    cluster_of, _ = kmeans_fit(X, m, seed, tag)
    typ = np.zeros(len(pool))
    for c in np.unique(cluster_of):
        members = np.flatnonzero(cluster_of == c)
        if len(members) == 1:
            typ[members] = np.inf
            continue
        d = euclidean_distances(X[members])
        k = min(k_typ, len(members) - 1)
        knn = np.sort(d, axis=1)[:, 1:k + 1]          # pula a distancia a si mesmo
        typ[members] = 1.0 / (knn.mean(axis=1) + 1e-12)
    return _one_per_cluster(pool, cluster_of, typ)
