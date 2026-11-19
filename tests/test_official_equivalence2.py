"""
Second equivalence test: CoreSet, DCoM, RIM, Grain and AGE against the ORIGINAL CODE.

  CoreSet  runs CoreSetMIPSampling.greedy_k_center (avihu111/TypiClust), which query() uses
  DCoM     runs the original DCoM class (same repository): binary-search deltas and picks
  RIM      runs the functions of the original RIM.py (zwt233/RIM) on our graph, with the loop of the script
  Grain    runs the logic of the original GRAIN(ball-D).py (zwt233/Grain) on our graph
  AGE      compares perc/percd and the choice with the code of train_entropy_density_graphcentral_ts.py

Here the ports break ties by the smallest index (rng=None), as the original does; in the
experiments ties are broken at random.

  OFFICIAL_REPOS=/path/with/TypiClust,RIM,Grain,AGE  python3 tests/test_official_equivalence2.py
"""

import ast
import os
import sys
import types

import numpy as np
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
REPOS = os.environ.get("OFFICIAL_REPOS")
if not REPOS:
    sys.exit("defina OFFICIAL_REPOS=/pasta/que/contem/TypiClust,RIM,Grain,AGE")
sys.path.insert(0, os.path.join(REPOS, "TypiClust", "deep-al"))
sys.modules.setdefault("faiss", types.SimpleNamespace())
if not hasattr(np, "float"):
    np.float = float          # o DCoM original usa np.float (removido do numpy novo)

import torch  # noqa: E402
import data  # noqa: E402
import coverage as cv  # noqa: E402
import official_ports as op  # noqa: E402
from gcn import fit  # noqa: E402
from graphs import build_edge_index  # noqa: E402
from protocol import fixed_test_split  # noqa: E402

DS, DESC = "flowers", "Resnet152"


def overlap(a, b):
    a, b = list(map(int, a)), list(map(int, b))
    return len(set(a) & set(b)) / max(len(a), 1)


def load():
    labels = data.load_labels(DS)
    X = data.load_features(DS, DESC)
    rks = data.load_rks(DS, DESC, L=41)
    ei = build_edge_index(rks, 40, "rec")
    pool, _ = fixed_test_split(labels, 0)
    return labels, X, rks, ei, np.sort(pool)


def extract_functions(path, names):
    """Executa so' as definicoes de funcao do script (o resto carrega o Cora)."""
    tree = ast.parse(open(path).read())
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                     type_ignores=[])
    return compile(mod, path, "exec")


# ------------------------------------------------------------------ CoreSet
def test_coreset(labels, X, pool, emb):
    from pycls.al.Sampling import CoreSetMIPSampling
    rng = np.random.default_rng(0)
    lab = np.sort(rng.choice(pool, 40, replace=False))
    unl = np.setdiff1d(pool, lab)
    q = 30
    cs = CoreSetMIPSampling.__new__(CoreSetMIPSampling)
    cs.cfg = types.SimpleNamespace(ACTIVE_LEARNING=types.SimpleNamespace(BUDGET_SIZE=q))
    cs.isMIP = False
    # greedy_k_center e' o que o query() oficial usa (optimal_greedy_k_center tem bug e esta' comentado)
    idx, _ = cs.greedy_k_center(emb[lab].astype(np.float32), emb[unl].astype(np.float32))
    official = unl[np.asarray(idx, dtype=int)]
    port = op.coreset_greedy(unl, q, lab, emb)
    print(f"CoreSet (embeddings da GCN): porte vs original = {overlap(port, official):.3f}, "
          f"mesma ordem: {list(port) == list(official)}")


# ------------------------------------------------------------------ DCoM
class AttrDict(dict):
    def __getattr__(self, k):
        v = self[k]
        return AttrDict(v) if isinstance(v, dict) else v


def test_dcom(labels, X, pool, logp_fn):
    import pycls.al.DCoM as off
    fn = op.l2_normalize(X)
    off.ds_utils.load_features = lambda *a, **k: fn
    _orig = off.DCoM.construct_graph_excluding_lSet
    # sem o bug do ultimo lote (o original so' percorre len // batch_size lotes)
    off.DCoM.construct_graph_excluding_lSet = lambda self, delta=None, batch_size=700: _orig(
        self, delta, batch_size=len(self.rel_features))
    cfg = AttrDict({"DATASET": {"NAME": "x"}, "RNG_SEED": 0,
                    "ACTIVE_LEARNING": {"K_LOGISTIC": op.DCOM_K_LOGISTIC, "A_LOGISTIC": op.DCOM_A_LOGISTIC,
                                        "DELTA_RESOLUTION": op.DCOM_DELTA_RESOLUTION}})
    delta0 = cv.probcover_delta(X, int(labels.max()) + 1, tag="dcomeq")
    m0, q = 30, 20

    # rodada inicial (sem modelo: competencia ~ 0)
    o = off.DCoM(cfg, np.array([], dtype=int), pool, m0, [str(delta0)] * m0, op.DCOM_MAX_DELTA)
    o.calculate_margin = lambda *a, **k: np.zeros(len(o.relevant_indices))
    off_init, _ = o.select_samples(None, None, None)
    st = op.DCoMState(pool, X, initial_delta=delta0)
    port_init, _, _ = st.select(m0)
    print(f"DCoM inicial: porte vs original = {overlap(port_init, off_init):.3f}")

    # busca binaria dos deltas: mesmos centros, mesmos pseudo-rotulos
    lab = np.asarray(off_init)
    logp = logp_fn(np.sort(lab))
    pseudo = logp.argmax(1)
    o2 = off.DCoM(cfg, lab, np.setdiff1d(pool, lab), m0, [str(delta0)] * m0, op.DCOM_MAX_DELTA)
    rel = o2.relevant_indices
    off_deltas = [float(d) for d in o2.new_centroids_deltas(labels[lab], pseudo[rel], m0, all_labels=labels[rel])]
    st2 = op.DCoMState(pool, X, initial_delta=delta0)
    st2.labeled, st2.deltas = [int(v) for v in lab], [delta0] * len(lab)
    st2.update_deltas(len(lab), labels, pseudo)
    same = np.allclose(off_deltas, st2.deltas)
    print(f"DCoM deltas da busca binaria: iguais ao original = {same} "
          f"(media original {np.mean(off_deltas):.3f}, porte {np.mean(st2.deltas):.3f})")

    # rodada com modelo
    o3 = off.DCoM(cfg, lab, np.setdiff1d(pool, lab), q, [str(d) for d in off_deltas], op.DCOM_MAX_DELTA)
    prob = np.sort(np.exp(logp[o3.relevant_indices]), axis=1)
    mr = -(prob[:, -1] - prob[:, -2])
    o3.calculate_margin = lambda *a, **k: (mr - mr.min()) / (mr.max() - mr.min())
    off_q, _ = o3.select_samples(None, None, None)
    port_q, comp, cov = st2.select(q, logp)
    print(f"DCoM rodada com modelo (competencia {comp}, cobertura {cov:.2f}): porte vs original = "
          f"{overlap(port_q, off_q):.3f}")


# ------------------------------------------------------------------ RIM
def test_rim(labels, ei, pool):
    n = len(labels)
    A = op._adj(ei, n) + sp.identity(n)
    Ar = sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
    A2 = np.asarray((Ar @ Ar).todense())
    for th in (0.0, op.rim_threshold_median(ei, n)):
        ns = {"np": np, "adj_matrix2": A2, "num_ones": np.ones(n), "th": th, "oracle_acc": 1.0,
              "num_class": int(labels.max()) + 1, "reliability_list": np.ones(n),
              "similarity_feature": np.ones((n, n)), "labels": list(labels)}
        exec(extract_functions(os.path.join(REPOS, "RIM", "examples", "RIM.py"),
                               {"get_reliable_score", "get_activated_node_dense",
                                "get_max_reliable_info_node_dense", "update_reliability"}), ns)
        # laco principal do RIM.py, sem a parte de treino
        activated, idx_train, train_class = np.ones(n), [], {}
        avail = list(pool)
        budget, batch = 60, 5
        while True:
            node, act_vec, num = ns["get_max_reliable_info_node_dense"](idx_train, avail, activated, train_class,
                                                                       labels)
            idx_train.append(node)
            avail.remove(node)
            train_class.setdefault(int(labels[node]), []).append(node)
            if len(idx_train) % batch == 0:
                activated = ns["update_reliability"](idx_train, train_class, labels, n)
            activated = activated - act_vec
            if len(idx_train) >= budget or num <= 0:
                break
        port = op.rim_select(pool, len(idx_train), ei, n, th, np.random.default_rng(0), tie_rng=None)
        print(f"RIM (th={th:.4f}): porte vs original = {overlap(port, idx_train):.3f}, "
              f"mesma ordem: {list(port) == idx_train} ({len(idx_train)} escolhas)")


# ------------------------------------------------------------------ Grain
def test_grain(X, ei, pool):
    n = len(X)
    A = op._adj(ei, n) + sp.identity(n)
    dinv = 1 / np.sqrt(np.asarray(A.sum(1)).ravel())
    adj = np.asarray((sp.diags(dinv) @ A @ sp.diags(dinv)).todense())
    adj_matrix2 = adj @ adj
    features_aax = adj_matrix2 @ X.astype(np.float64)
    sq = (features_aax ** 2).sum(1)
    distance_aax = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * features_aax @ features_aax.T, 0))
    np.fill_diagonal(distance_aax, 0)
    distance_aax = (distance_aax - distance_aax.min()) / (distance_aax.max() - distance_aax.min())
    # logica literal do GRAIN(ball-D).py
    balls = (distance_aax <= op.GRAIN_RADIUS).astype(float)
    balls_dict = {}
    for node in pool:
        neighbors_tmp = ((adj_matrix2[[node]].sum(axis=0) != 0) + 0)[:, np.newaxis]
        dot_result = np.matmul(balls, neighbors_tmp).T
        balls_dict[node] = set(np.flatnonzero(dot_result[0] != 0).tolist())
    covered, idx_train, avail = set(), [], list(pool)
    budget = 60
    while True:
        ball_num_max, node_max = 0, 0
        for node in avail:
            t = len(covered.union(balls_dict[node]))
            if t > ball_num_max:
                ball_num_max, node_max = t, node
        idx_train.append(node_max)
        avail.remove(node_max)
        covered = covered.union(balls_dict[node_max])
        if len(idx_train) >= budget or n - ball_num_max == 0:
            break
    port = op.grain_select(pool, len(idx_train), X, ei, n, np.random.default_rng(0), tie_rng=None)
    print(f"Grain: porte vs original = {overlap(port, idx_train):.3f}, mesma ordem: "
          f"{list(port) == idx_train} ({len(idx_train)} escolhas)")


# ------------------------------------------------------------------ AGE
def test_age():
    def perc(inp, k):
        return sum([1 if i else 0 for i in inp < inp[k]]) / float(len(inp))

    def percd(inp, k):
        return sum([1 if i else 0 for i in inp > inp[k]]) / float(len(inp))
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = np.round(rng.random(300), 2)          # com empates
        a = np.array([perc(v, k) for k in range(len(v))])
        b = np.array([percd(v, k) for k in range(len(v))])
        assert np.allclose(a, op._perc(v)) and np.allclose(b, op._percd(v))
    print("AGE: perc/percd vetorizados == funcoes originais (com empates)")


if __name__ == "__main__":
    labels, X, rks, ei, pool = load()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, eid = torch.from_numpy(X).to(dev), ei.to(dev)

    def logp_fn(lab):
        return fit("gcn_net", x, eid, labels, lab, 0)[0]
    _, hidden = fit("gcn_net", x, eid, labels, np.sort(np.random.default_rng(0).choice(pool, 40, replace=False)), 0)
    test_age()
    test_coreset(labels, X, pool, hidden)
    test_rim(labels, ei, pool)
    test_grain(X, ei, pool)
    test_dcom(labels, X, pool, logp_fn)
