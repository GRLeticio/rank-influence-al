"""
Experiment 08 -- computational cost: time of the selection ONLY for each method (no GCN training).

Same inputs for all methods: dataset/descriptor, split 0, 10% budget.
  cold start (no model): select B/2 images
  query (with model):    select B/10 images (one round), given the B/2 labels of MaxHerding
                         and a GCN-Net trained with them
Each measurement is repeated (--reps) and the median is reported. The machine is recorded.

Usage (from the repository root):
  python3 experiments/08_cost/run.py                       # CUB and Corel5k, ResNet152
  python3 experiments/08_cost/run.py --datasets flowers --reps 1
"""

import argparse
import csv
import os
import platform
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import coverage as cv
import data
import official_ports as op
import selection
from active import pick_influence_variant, reverse_matrix, uncertainty_prob
from gcn import fit
from graphs import build_edge_index
from protocol import fixed_test_split

import importlib.util
_spec = importlib.util.spec_from_file_location("run05", os.path.join(ROOT, "experiments", "05_al_confirm", "run.py"))
run05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run05)


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cub200", "corel5k"])
    p.add_argument("--descriptor", default="Resnet152")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "08_cost"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    machine = f"{platform.node()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    rows = []
    for ds in a.datasets:
        labels = data.load_labels(ds)
        n, C = len(labels), int(labels.max()) + 1
        feats = data.load_features(ds, a.descriptor)
        rks = data.load_rks(ds, a.descriptor, L=max(run05.GRAPH_K, 41))
        ei = build_edge_index(rks, run05.GRAPH_K, "rec")
        pool, test = fixed_test_split(labels, 0)
        B = int(round(0.10 * n))
        m0, q = B // 2, B // 10
        rng = lambda: np.random.default_rng(0)   # noqa: E731
        delta = cv.probcover_delta(feats, C, tag=f"cost|{ds}|{a.descriptor}")
        print(f"\n{ds}: n={n}, pool={len(pool)}, peca 1 escolhe {m0}, peca 2 escolhe {q}  [{machine}]", flush=True)

        p1 = {
            "sorteio": lambda: rng().permutation(pool)[:m0],
            "TypiClust (oficial)": lambda: op.typiclust_official(pool, m0, feats, seed=0),
            "ProbCover (oficial)": lambda: op.probcover_official(pool, m0, feats, delta, rng=rng()),
            "MaxHerding (fiel)": lambda: op.maxherding_paper(pool, m0, feats, rng=rng()),
            "Rank-ProbCover (nosso)": lambda: run05._ordered_rank_probcover(pool, m0, rks),
            "RIM": lambda: op.rim_select(pool, m0, ei, n, op.rim_threshold_median(ei, n), rng(), tie_rng=rng()),
            "Grain": lambda: op.grain_select(pool, m0, feats, ei, n, rng(), tie_rng=rng()),
            "DCoM (inicial)": lambda: op.DCoMState(pool, feats, delta, rng=rng()).select(m0)[0],
        }
        for name, fn in p1.items():
            t, _ = timed(fn, a.reps)
            rows.append({"dataset": ds, "peca": 1, "metodo": name, "n_escolhas": m0, "segundos": f"{t:.3f}", "maquina": machine})
            print(f"  peca 1  {name:24s} {t:8.2f} s", flush=True)

        # entradas da peca 2: B/2 rotulados do MaxHerding + uma GCN-Net treinada
        lab = np.sort(op.maxherding_paper(pool, m0, feats, rng=rng()))
        x = torch.from_numpy(feats).to(dev)
        logp, hidden = fit("gcn_net", x, ei.to(dev), labels, lab, 0)
        cand = np.setdiff1d(pool, lab)
        R = reverse_matrix(rks, run05.INFL_K, run05.INFL_P)
        centrality = op.pagerank_normalized(ei, n)
        st = op.DCoMState(pool, feats, delta, rng=rng())
        st.select(m0)                                      # deixa o estado do DCoM como depois da peca 1

        def dcom_round():
            s2 = op.DCoMState(pool, feats, delta, rng=rng())
            s2.labeled, s2.deltas = list(st.labeled), list(st.deltas)
            s2.update_deltas(len(s2.labeled), labels, logp.argmax(1))
            return s2.select(q, logp)[0]

        p2 = {
            "margem": lambda: cand[np.argsort(-uncertainty_prob(logp)[cand])[:q]],
            "information density": lambda: op.info_density_query(logp, cand, q, feats, unlabeled=cand),
            "CoreSet (oficial, embeddings)": lambda: op.coreset_greedy(cand, q, lab, hidden),
            "AGE (oficial)": lambda: op.age_query(logp, cand, q, C, centrality, 0, rng()),
            "DCoM (rodada, com deltas)": dcom_round,
            "influencia por rank (nossa)": lambda: pick_influence_variant(logp, cand, q, rng(), lab, R),
        }
        for name, fn in p2.items():
            t, _ = timed(fn, a.reps)
            rows.append({"dataset": ds, "peca": 2, "metodo": name, "n_escolhas": q, "segundos": f"{t:.3f}", "maquina": machine})
            print(f"  peca 2  {name:30s} {t:8.2f} s", flush=True)
        t, _ = timed(lambda: fit("gcn_net", x, ei.to(dev), labels, lab, 0), 1)
        rows.append({"dataset": ds, "peca": 0, "metodo": "(referencia) 1 treino da GCN-Net", "n_escolhas": 0,
                     "segundos": f"{t:.3f}", "maquina": machine})
        print(f"  (referencia) 1 treino da GCN-Net: {t:.2f} s", flush=True)
        del x
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
