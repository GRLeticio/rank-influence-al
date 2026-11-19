"""
Tests OUR method against slow and obvious versions (plain loops), written again
from the definition, without reusing the vectorized code:

  1. the 10-flower example worked out by hand (priorities and choice)
  2. greedy influence: the fast version (active.pick_rank_influence_greedy, used in exp 05)
     selects the SAME images, in the SAME order, as the slow version -- random cases and
     real data (Flowers and Corel5k, with a really trained GCN)
  3. Rank-ProbCover: same (exp 05 _ordered_rank_probcover vs slow version)

Run from the repository root:  python3 tests/test_influence.py
"""

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import active  # noqa: E402

_spec = importlib.util.spec_from_file_location("run05", os.path.join(ROOT, "experiments", "05_al_confirm", "run.py"))
run05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run05)


# ------------------------------------------------------------------ versoes lentas

def slow_influence_greedy(unc0, rks, k, p, cand, q, labeled, noise):
    """Definicao literal: prioridade(v) = u(v) * soma_{w : v na lista top-k de w} p^(pos-1) * u(w);
    escolhe o maximo; desconta u(w) *= 1 - p^(pos-1) pra quem tem a escolhida; u(escolhida) = 0."""
    n = len(unc0)
    u = np.array(unc0, dtype=float)
    u[list(labeled)] = 0.0
    lists = [list(rks[w][1:k + 1]) for w in range(n)]
    avail = set(int(c) for c in cand)
    out = []
    for _ in range(q):
        best, best_s = None, -np.inf
        for v in sorted(avail):
            infl = 0.0
            for w in range(n):
                if v in lists[w]:
                    pos = lists[w].index(v)            # 0 = primeira posicao
                    infl += (p ** pos) * u[w]
            s = u[v] * infl + noise[v]
            if s > best_s:
                best, best_s = v, s
        out.append(best)
        avail.discard(best)
        for w in range(n):
            if best in lists[w]:
                u[w] *= 1.0 - p ** lists[w].index(best)
        u[best] = 0.0
    return out


def slow_rank_probcover(pool, m, rks, k, noise):
    n = len(rks)
    covered = set()
    avail = set(int(v) for v in pool)
    out = []
    for _ in range(m):
        best, best_s = None, -np.inf
        for v in sorted(avail):
            ball = set(int(x) for x in rks[v][:k + 1])
            s = len(ball - covered) + noise[v]
            if s > best_s:
                best, best_s = v, s
        out.append(best)
        avail.discard(best)
        covered |= set(int(x) for x in rks[best][:k + 1])
    return out


# ------------------------------------------------------------------ 1. exemplo na mao

def test_hand_example():
    names = "ABCDEFGHIJ"
    ix = {c: i for i, c in enumerate(names)}
    lst = {"A": "BCD", "B": "ACE", "C": "ABD", "D": "CAE", "E": "BDF", "F": "GHE",
           "G": "FHI", "H": "GFI", "I": "HGF", "J": "EIA"}
    rks = np.array([[ix[c]] + [ix[x] for x in lst[c]] for c in names])
    doubt = np.array([0.1, 0.1, 0.1, 0.2, 0.9, 0.3, 0.1, 0.1, 0.2, 0.95])
    R = active.reverse_matrix(rks, 3, 0.9)
    infl = R.T @ doubt
    prio = doubt * infl
    # contas feitas na mao na conversa: influencia(E) = 0.81*0.1 + 0.81*0.2 + 0.81*0.3 + 1*0.95
    assert np.isclose(infl[ix["E"]], 0.81 * 0.1 + 0.81 * 0.2 + 0.81 * 0.3 + 0.95)
    assert np.isclose(prio[ix["E"]], 0.9 * 1.436)
    assert infl[ix["J"]] == 0                       # ninguem tem J na lista
    cand = np.array([ix[c] for c in "BCDEGHIJ"])     # A e F ja rotulados
    pick = active.pick_rank_influence_greedy(logp=None, cand=cand, q=1, rng=np.random.default_rng(0),
                                             labeled=[ix["A"], ix["F"]], R=R,
                                             unc_fn=lambda _: doubt.copy())
    assert names[pick[0]] == "E", names[pick[0]]
    print("ok exemplo das 10 flores: influencia(E) = 1.436, prioridade(E) = 1.292, escolhe E (a margem escolheria J)")


# ------------------------------------------------------------------ 2. influencia: rapida vs lenta

def compare_influence(unc, rks, cand, q, labeled, seed, k=run05.INFL_K, p=run05.INFL_P):
    R = active.reverse_matrix(rks, k, p)
    fast = active.pick_rank_influence_greedy(logp=None, cand=np.asarray(cand), q=q, rng=np.random.default_rng(seed),
                                             labeled=list(labeled), R=R, unc_fn=lambda _: unc.copy())
    noise = np.random.default_rng(seed).random(len(unc)) * 1e-9   # o mesmo ruido de desempate da rapida
    slow = slow_influence_greedy(unc, rks, k, p, cand, q, labeled, noise)
    return [int(v) for v in fast], slow


def test_influence_random():
    rng = np.random.default_rng(1)
    for trial in range(30):
        n = int(rng.integers(30, 80))
        k = int(rng.integers(2, 8))
        rks = np.stack([np.concatenate([[i], rng.permutation(np.setdiff1d(np.arange(n), [i]))[:k + 3]])
                        for i in range(n)])
        unc = rng.random(n)
        if trial % 3 == 0:
            unc = np.round(unc, 1)                      # muitos empates de proposito
        labeled = rng.choice(n, int(rng.integers(0, 6)), replace=False)
        cand = np.setdiff1d(np.arange(n), labeled)
        q = int(rng.integers(1, 10))
        p = float(rng.choice([0.8, 0.9, 1.0]))
        fast, slow = compare_influence(unc, rks, cand, q, labeled, seed=trial, k=k, p=p)
        assert fast == slow, (trial, fast, slow)
    print("ok influencia gulosa: rapida == lenta em 30 casos aleatorios (com empates, p = 0.8 / 0.9 / 1.0)")


def test_influence_real():
    import torch
    import data
    from gcn import fit
    from graphs import build_edge_index
    from protocol import fixed_test_split
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds, n_sub in [("flowers", None), ("corel5k", 1200)]:
        labels = data.load_labels(ds)
        rks = data.load_rks(ds, "Resnet152", L=41)
        x = torch.from_numpy(data.load_features(ds, "Resnet152")).to(dev)
        ei = build_edge_index(rks, 40, "rec").to(dev)
        pool, _ = fixed_test_split(labels, 0)
        lab = np.sort(np.random.default_rng(0).choice(pool, int(0.05 * len(labels)), replace=False))
        logp, _ = fit("gcn_sgc", x, ei, labels, lab, 0)
        unc = active.uncertainty_prob(logp)
        cand = np.setdiff1d(pool, lab)
        if n_sub:                                      # a lenta e' O(q * n^2): limita os candidatos no Corel5k
            cand = np.sort(np.random.default_rng(1).choice(cand, n_sub, replace=False))
        fast, slow = compare_influence(unc, rks, cand, 15, lab, seed=7)
        assert fast == slow, (ds, fast, slow)
        print(f"ok influencia gulosa em dados reais ({ds}/ResNet152, GCN treinada, 15 escolhas): rapida == lenta")


# ------------------------------------------------------------------ 3. Rank-ProbCover: rapida vs lenta

def test_rank_probcover():
    import data
    from protocol import fixed_test_split
    for ds in ["flowers", "corel5k"]:
        labels = data.load_labels(ds)
        rks = data.load_rks(ds, "Resnet152", L=41)
        pool, _ = fixed_test_split(labels, 0)
        m = 40
        fast = [int(v) for v in run05._ordered_rank_probcover(pool, m, rks)]
        noise = np.random.default_rng(0).random(len(labels)) * 1e-6   # o mesmo da rapida
        slow = slow_rank_probcover(pool, m, rks, run05.COVER_K, noise)
        assert fast == slow, (ds, fast[:10], slow[:10])
        print(f"ok Rank-ProbCover ({ds}/ResNet152, {m} escolhas): rapida == lenta")


def test_variant_equals_full():
    """A variante parametrizada do exp 07, com tudo ligado, e' a mesma influencia do exp 05."""
    rng = np.random.default_rng(3)
    for t in range(40):
        n = int(rng.integers(30, 90))
        k = int(rng.integers(2, 8))
        p = float(rng.choice([0.8, 0.9, 1.0]))
        rks = np.stack([np.concatenate([[i], rng.permutation(np.setdiff1d(np.arange(n), [i]))[:k + 2]])
                        for i in range(n)])
        logp = np.log(rng.dirichlet(np.ones(5) * (0.3 if t % 2 else 2), size=n))
        lab = rng.choice(n, int(rng.integers(0, 6)), replace=False)
        cand = np.setdiff1d(np.arange(n), lab)
        q = int(rng.integers(1, 10))
        R = active.reverse_matrix(rks, k, p)
        a = active.pick_rank_influence_greedy(logp=logp, cand=cand, q=q, rng=np.random.default_rng(t), labeled=lab,
                                              R=R, unc_fn=active.uncertainty_prob)
        b = active.pick_influence_variant(logp, cand, q, np.random.default_rng(t), lab, R)
        assert list(a) == list(b), (t, a, b)
    print("ok variante do exp 07 com tudo ligado == influencia do exp 05 (40 casos)")


if __name__ == "__main__":
    test_variant_equals_full()
    test_hand_example()
    test_influence_random()
    test_rank_probcover()
    test_influence_real()
