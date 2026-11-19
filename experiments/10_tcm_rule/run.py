"""
Experiment 10 -- TCM with rule 1 of its authors. EXPLORATORY.

  tcm_rule1:                 official TypiClust on min(B, 3C) images; the rest by MARGIN in rounds of C images.
  typiclust_rule1_influence: same scheme, with our INFLUENCE instead of margin (--query influence).

Same seeds, splits and GCNs as exp 06; reuses ModelFree and SafeFit without modifying exp 06.

Usage (from the repository root):
  python3 experiments/10_tcm_rule/run.py --datasets flowers --descriptors Resnet152 --reranks none
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import coverage as cv  # noqa: E402
import data  # noqa: E402
from active import reverse_matrix  # noqa: E402
from graphs import build_edge_index  # noqa: E402
from protocol import fixed_test_split  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run05 = _load("run05", "experiments/05_al_confirm/run.py")
run06 = _load("run06", "experiments/06_al_baselines/run.py")
PIPES = {"margin": "tcm_rule1", "influence": "typiclust_rule1_influence"}
STEPS = 3


def schedule(B, C):
    """(rotulos do TypiClust, tamanhos das rodadas de margem)."""
    m0 = min(B, STEPS * C)
    rest = B - m0
    sizes = [C] * (rest // C) + ([rest % C] if rest % C else [])
    return m0, sizes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["flowers", "corel5k", "cub200"])
    p.add_argument("--descriptors", nargs="+", default=["Resnet152", "vitb16", "swintf"])
    p.add_argument("--reranks", nargs="+", default=["none"])
    p.add_argument("--gcn-types", nargs="+", default=["gcn_sgc", "gcn_net"])
    p.add_argument("--budgets", nargs="+", type=float, default=list(run05.BUDGETS))
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--query", choices=["margin", "influence"], default="margin")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "10_tcm_rule"))
    a = p.parse_args()
    PIPE = PIPES[a.query]
    os.makedirs(a.out, exist_ok=True)
    csv_path, log_path = os.path.join(a.out, "results.csv"), os.path.join(a.out, "log.txt")
    done = run05.completed_keys(csv_path)
    log = run05.log
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} args={vars(a)}", log_path)
    seed = 0
    for dataset in a.datasets:
        labels = data.load_labels(dataset)
        n, C = len(labels), int(labels.max()) + 1
        kw = data.gcn_params(dataset)
        budgets = {b: int(round(b * n)) for b in a.budgets}
        for desc in a.descriptors:
            feats = data.load_features(dataset, desc)
            x = torch.from_numpy(feats).to(device)
            delta = cv.probcover_delta(feats, C, seed=0, tag=f"{dataset}|{desc}")
            for rr in a.reranks:
                rerank = "" if rr == "none" else rr
                rks = data.load_rks(dataset, desc, L=max(run05.GRAPH_K, run05.INFL_K + 1), rerank=rerank)
                ei = build_edge_index(rks, run05.GRAPH_K, "rec").to(device)
                R = reverse_matrix(rks, run05.INFL_K, run05.INFL_P)
                safe_fit = run05.SafeFit(x, ei, log_path)
                for sp in range(a.splits):
                    pool, test = fixed_test_split(labels, sp)
                    mf = run06.ModelFree(pool, feats, ei, n, C, 30_000 + 100 * sp, delta)
                    for gcn_type in a.gcn_types:
                        base = f"{dataset}|{desc}|{rr}|{gcn_type}|{sp}|{seed}"
                        todo = [b for b in budgets if f"{base}|{b}|{PIPE}" not in done]
                        if not todo:
                            continue
                        t_unit, rows = time.time(), []
                        for b in todo:
                            B = budgets[b]
                            t0 = time.time()
                            rng = np.random.default_rng(20_000 + 100 * sp + seed)
                            m0, sizes = schedule(B, C)
                            lab = mf.get("typiclust_off", m0, seed)
                            logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                            acc0 = (logp.argmax(1)[test] == labels[test]).mean() * 100
                            hits = asked = 0
                            for q in sizes:
                                cand = np.setdiff1d(pool, lab)
                                pick = np.asarray(run05.query(a.query, logp, cand, q, rng, lab, feats, R))
                                assert len(np.unique(pick)) == q and np.isin(pick, cand).all()
                                hits += int((logp.argmax(1)[pick] != labels[pick]).sum())
                                asked += q
                                lab = np.sort(np.concatenate([lab, pick]))
                                logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                            assert len(lab) == B and not np.intersect1d(lab, test).size
                            acc = (logp.argmax(1)[test] == labels[test]).mean() * 100
                            rows.append({"key": f"{base}|{b}|{PIPE}", "dataset": dataset, "descriptor": desc,
                                         "rerank": rr, "gcn_type": gcn_type, "split": sp, "seed": seed,
                                         "budget": b, "n_labeled": B, "pipeline": PIPE,
                                         "acc": f"{acc:.4f}", "acc_initial": f"{acc0:.4f}",
                                         "hit_rate": f"{hits/asked:.4f}" if asked else "",
                                         "coverage": f"{len(np.unique(labels[lab]))/C:.4f}",
                                         "seconds": f"{time.time()-t0:.1f}"})
                        run05.append_rows(csv_path, rows)
                        log(f"{base}: {len(rows)} linhas ({time.time()-t_unit:.0f}s)", log_path)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                del ei, safe_fit
            del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log("fim", log_path)


if __name__ == "__main__":
    main()
