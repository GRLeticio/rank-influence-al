"""
Experiment 07 -- ablation of the components of the influence + sensitivity (EXPLORATORY).

Fixed cold start: MaxHerding as in the paper (official_ports.maxherding_paper), B/2 labels.
Query: the full influence and variants, each removing or changing ONE thing.

  full            full influence (k=20, p=0.9, 5 rounds, with discount)
  margin          margin instead of influence (reference)
  A1_nodiscount   no greedy discount (takes the q largest at once)
  A2_nopos        no position weight (p = 1: being 1st or 20th counts the same)
  A3_noself       no doubt of the image itself (priority = influence only)
  A4_nonbr        no doubt of the neighbors (counts 1 per reverse neighbor: "popularity")
  A5_inductive    ignores the doubt of TEST images in the influence (inductive version)
  S_k10, S_k40    influence k = 10 / 40
  S_p05, S_p07, S_p08   position weight p = 0.5 / 0.7 / 0.8 (p = 1.0 is A2)
  S_r1, S_r10     1 round (all at once) / 10 rounds

Original lists: all variants. LHRR lists: full, margin, A2_nopos (does the order of the list
matter more when the list is better?).

Same protocol as exp 05/06 (fixed 20% test set, 10 splits, budgets of 2/5/10%), on a subset:
3 datasets x ResNet152 and ViT-B/16 x GCN-Net.

Usage (from the repository root):
  python3 experiments/07_ablation_sensitivity/run.py --smoke
  python3 experiments/07_ablation_sensitivity/run.py --datasets flowers --descriptors Resnet152 --reranks none
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

import data
import official_ports as op
from active import pick_influence_variant, reverse_matrix, uncertainty_prob
from graphs import build_edge_index
from protocol import fixed_test_split

_spec = importlib.util.spec_from_file_location("run05", os.path.join(ROOT, "experiments", "05_al_confirm", "run.py"))
run05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run05)

K0, P0, ROUNDS0 = run05.INFL_K, run05.INFL_P, run05.ROUNDS
VARIANTS = {
    "full": {},
    "margin": {"margin": True},
    "A1_nodiscount": {"discount": False},
    "A2_nopos": {"p": 1.0},
    "A3_noself": {"self_doubt": False},
    "A4_nonbr": {"nbr_doubt": False},
    "A5_inductive": {"inductive": True},
    "S_k10": {"k": 10},
    "S_k40": {"k": 40},
    "S_p05": {"p": 0.5},
    "S_p07": {"p": 0.7},
    "S_p08": {"p": 0.8},
    "S_r1": {"rounds": 1},
    "S_r10": {"rounds": 10},
}
RERANK_VARIANTS = ["full", "margin", "A2_nopos"]
FIELDS = run05.FIELDS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["flowers", "corel5k", "cub200"])
    p.add_argument("--descriptors", nargs="+", default=["Resnet152", "vitb16"])
    p.add_argument("--reranks", nargs="+", default=["none", "LHRR"])
    p.add_argument("--gcn-types", nargs="+", default=["gcn_net"])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--budgets", nargs="+", type=float, default=list(run05.BUDGETS))
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "07_ablation_sensitivity"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.datasets, a.descriptors, a.reranks, a.gcn_types = ["flowers"], ["Resnet152"], ["none"], ["gcn_sgc"]
        a.splits, a.budgets = 1, [0.10]
        a.out = os.path.join(ROOT, "results", "07_ablation_sensitivity_smoke")
    return a


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "results.csv")
    log_path = os.path.join(a.out, "log.txt")
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
            for rr in a.reranks:
                rerank = "" if rr == "none" else rr
                rks = data.load_rks(dataset, desc, L=max(run05.GRAPH_K, 41), rerank=rerank)
                ei = build_edge_index(rks, run05.GRAPH_K, "rec").to(device)
                Rs = {}

                def R_of(k, p):
                    if (k, p) not in Rs:
                        Rs[(k, p)] = reverse_matrix(rks, k, p)
                    return Rs[(k, p)]
                safe_fit = run05.SafeFit(x, ei, log_path)
                variants = [v for v in a.variants if rr == "none" or v in RERANK_VARIANTS]
                for sp in range(a.splits):
                    pool, test = fixed_test_split(labels, sp)
                    # peca 1: MaxHerding fiel, uma vez por divisao (guloso: o menor e' prefixo do maior)
                    mh = None
                    for gcn_type in a.gcn_types:
                        base = f"{dataset}|{desc}|{rr}|{gcn_type}|{sp}|{seed}"
                        todo = [(b, v) for b in budgets for v in variants if f"{base}|{b}|{v}" not in done]
                        if not todo:
                            continue
                        if mh is None:
                            mh = op.maxherding_paper(pool, max(budgets.values()) // 2, feats,
                                                     rng=np.random.default_rng(30_000 + 100 * sp + seed))
                        t_unit = time.time()
                        rows, init_fit = [], {}
                        for b, vname in todo:
                            cfg = VARIANTS[vname]
                            B = budgets[b]
                            m0 = B // 2
                            t0 = time.time()
                            lab = np.sort(mh[:m0])
                            if m0 not in init_fit:
                                init_fit[m0] = safe_fit(gcn_type, labels, lab, seed, **kw)
                            logp = init_fit[m0]
                            acc0 = (logp.argmax(1)[test] == labels[test]).mean() * 100
                            rng = np.random.default_rng(20_000 + 100 * sp + seed)
                            rounds = cfg.get("rounds", ROUNDS0)
                            R = R_of(cfg.get("k", K0), cfg.get("p", P0))
                            hits = asked = 0
                            for q in [len(c) for c in np.array_split(np.arange(B - m0), rounds)]:
                                if q == 0:
                                    continue
                                cand = np.setdiff1d(pool, lab)
                                if cfg.get("margin"):
                                    u = uncertainty_prob(logp)[cand]
                                    pick = cand[np.lexsort((rng.random(len(cand)), -u))[:q]]
                                else:
                                    pick = pick_influence_variant(
                                        logp, cand, q, rng, lab, R,
                                        self_doubt=cfg.get("self_doubt", True), nbr_doubt=cfg.get("nbr_doubt", True),
                                        discount=cfg.get("discount", True),
                                        ignore=test if cfg.get("inductive") else None)
                                pick = np.asarray(pick)
                                assert len(np.unique(pick)) == q and np.isin(pick, cand).all()
                                hits += int((logp.argmax(1)[pick] != labels[pick]).sum())
                                asked += q
                                lab = np.sort(np.concatenate([lab, pick]))
                                logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                            assert len(lab) == B and not np.intersect1d(lab, test).size
                            acc = (logp.argmax(1)[test] == labels[test]).mean() * 100
                            rows.append({"key": f"{base}|{b}|{vname}", "dataset": dataset, "descriptor": desc,
                                         "rerank": rr, "gcn_type": gcn_type, "split": sp, "seed": seed,
                                         "budget": b, "n_labeled": B, "pipeline": vname,
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
