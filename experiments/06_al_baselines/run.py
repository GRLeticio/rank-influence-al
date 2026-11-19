"""
Experiment 06 -- official baselines, complementing exp 05.

Same protocol, splits, seeds and GCNs as exp 05: the rows written here pair with
those of exp 05 (results/05_al_confirm/results.csv) by the key
dataset x descriptor x list x GCN x split x seed x budget.
Exp 05 is not modified.

Pipelines (src/official_ports.py, ported from the authors' code):
  typiclust_off        official TypiClust over the whole budget
  tcm_off              official TypiClust (B/2) + margin (B/2)
  probcover_off        official ProbCover over the whole budget (purity delta)
  dcom                 official DCoM: coverage start + rounds with updated deltas
  age                  random (B/2) + AGE (B/2)
  info_density         random (B/2) + information density (B/2)
  rim_t0, rim_tmed     RIM without oracle noise, th = 0 and th = median (whole budget)
  grain                Grain (ball-D) over the whole budget
  typiclust_off_influence  official TypiClust (B/2) + our influence (B/2) -- plug-in
  maxherding_off       MaxHerding as in the paper (sigma = 1, full kernel, pool only)
  maxherding_off_margin    MaxHerding (B/2) + margin (B/2)
  maxherding_off_influence MaxHerding (B/2) + our influence (B/2)
  coreset_off          random (B/2) + CoreSet on the GCN representations (hidden layer),
                       as in the official code (which uses the penultimate layer of the classifier)

Usage (from the repository root):
  python3 experiments/06_al_baselines/run.py --smoke
  python3 experiments/06_al_baselines/run.py --datasets flowers --descriptors vitb16 --reranks none
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

import coverage as cv
import data
import official_ports as op
from active import reverse_matrix
from gcn import fit
from graphs import build_edge_index
from protocol import fixed_test_split

# reaproveita do exp 05 (sem alterar): constantes congeladas, SafeFit, consultas, log/CSV
_spec = importlib.util.spec_from_file_location("run05", os.path.join(ROOT, "experiments", "05_al_confirm", "run.py"))
run05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run05)

PIPELINES = ["typiclust_off", "tcm_off", "probcover_off", "dcom", "age", "info_density",
             "rim_t0", "rim_tmed", "grain", "typiclust_off_influence",
             "maxherding_off", "maxherding_off_margin", "maxherding_off_influence", "coreset_off"]
FULL_BUDGET = {"typiclust_off", "probcover_off", "rim_t0", "rim_tmed", "grain", "maxherding_off"}
FIELDS = run05.FIELDS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["flowers", "corel5k", "cub200"])
    p.add_argument("--descriptors", nargs="+", default=["Resnet152", "vitb16", "swintf"])
    p.add_argument("--reranks", nargs="+", default=["none"])
    p.add_argument("--gcn-types", nargs="+", default=["gcn_sgc", "gcn_net", "gcn_appnpnet"])
    p.add_argument("--pipelines", nargs="+", default=PIPELINES)
    p.add_argument("--budgets", nargs="+", type=float, default=list(run05.BUDGETS))
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "06_al_baselines"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.datasets, a.descriptors, a.gcn_types = ["flowers"], ["Resnet152"], ["gcn_sgc"]
        a.splits, a.budgets = 1, [0.10]
        a.out = os.path.join(ROOT, "results", "06_al_baselines_smoke")
    return a


class ModelFree:
    """Selecoes sem modelo (dependem so' da divisao), com cache por (tipo, m, semente)."""

    def __init__(self, pool, feats, ei, n, C, seed_base, delta):
        self.pool, self.feats, self.ei, self.n, self.C = pool, feats, ei, n, C
        self.seed_base, self.delta = seed_base, delta
        self.cache = {}
        self._th = None

    def get(self, kind, m, seed):
        key = (kind, m, seed)
        if key not in self.cache:
            rng = np.random.default_rng(self.seed_base + seed)
            if kind == "typiclust_off":
                sel = op.typiclust_official(self.pool, m, self.feats, seed=seed)
            elif kind == "probcover_off":
                sel = op.probcover_official(self.pool, m, self.feats, self.delta, rng=rng)
            elif kind == "rim_t0":
                sel = op.rim_select(self.pool, m, self.ei, self.n, 0.0, rng, tie_rng=rng)
            elif kind == "rim_tmed":
                if self._th is None:
                    self._th = op.rim_threshold_median(self.ei, self.n)
                sel = op.rim_select(self.pool, m, self.ei, self.n, self._th, rng, tie_rng=rng)
            elif kind == "grain":
                sel = op.grain_select(self.pool, m, self.feats, self.ei, self.n, rng, tie_rng=rng)
            elif kind == "maxherding_off":
                sel = op.maxherding_paper(self.pool, m, self.feats, rng=rng)
            elif kind == "random":
                sel = np.random.default_rng(10_000 + seed).permutation(self.pool)[:m]
            else:
                raise ValueError(kind)
            self.cache[key] = np.sort(np.asarray(sel))
        out = self.cache[key]
        assert len(out) == m and len(np.unique(out)) == m and np.isin(out, self.pool).all()
        return out


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "results.csv")
    log_path = os.path.join(a.out, "log.txt")
    done = run05.completed_keys(csv_path)
    log = run05.log
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} args={vars(a)}", log_path)

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
                centrality = op.pagerank_normalized(ei, n)
                safe_fit = run05.SafeFit(x, ei, log_path)
                for sp in range(a.splits):
                    pool, test = fixed_test_split(labels, sp)
                    mf = ModelFree(pool, feats, ei, n, C, 30_000 + 100 * sp, delta)
                    for gcn_type in a.gcn_types:
                        for seed in range(a.seeds):
                            base = f"{dataset}|{desc}|{rr}|{gcn_type}|{sp}|{seed}"
                            todo = [(b, pl) for b in budgets for pl in a.pipelines if f"{base}|{b}|{pl}" not in done]
                            if not todo:
                                continue
                            t_unit = time.time()
                            rows = []
                            for b, pl in todo:
                                B = budgets[b]
                                t0 = time.time()
                                rng = np.random.default_rng(20_000 + 100 * sp + seed)
                                hits = asked = 0
                                if pl in FULL_BUDGET:
                                    lab = mf.get(pl, B, seed)
                                    logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                                    acc0 = None
                                elif pl == "dcom":
                                    st = op.DCoMState(pool, feats, initial_delta=delta,
                                                      rng=np.random.default_rng(40_000 + 100 * sp + seed))
                                    lab, _, _ = st.select(B // 2)
                                    lab = np.sort(lab)
                                    logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                                    acc0 = (logp.argmax(1)[test] == labels[test]).mean() * 100
                                    n_new = len(lab)
                                    for q in [len(c) for c in np.array_split(np.arange(B - B // 2), run05.ROUNDS)]:
                                        if q == 0:
                                            continue
                                        st.update_deltas(n_new, labels, logp.argmax(1))
                                        pick, _, _ = st.select(q, logp)
                                        hits += int((logp.argmax(1)[pick] != labels[pick]).sum())
                                        asked += q
                                        n_new = q
                                        lab = np.sort(np.concatenate([lab, pick]))
                                        logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                                else:
                                    init_kind = {"tcm_off": "typiclust_off", "typiclust_off_influence": "typiclust_off",
                                                 "age": "random", "info_density": "random",
                                                 "maxherding_off_margin": "maxherding_off",
                                                 "maxherding_off_influence": "maxherding_off",
                                                 "coreset_off": "random"}[pl]
                                    lab = mf.get(init_kind, B // 2, seed)
                                    if pl == "coreset_off":
                                        logp, hidden = fit(gcn_type, x, ei, labels, lab, seed, **kw)
                                    else:
                                        logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                                    acc0 = (logp.argmax(1)[test] == labels[test]).mean() * 100
                                    for r, q in enumerate([len(c) for c in np.array_split(np.arange(B - B // 2), run05.ROUNDS)]):
                                        if q == 0:
                                            continue
                                        cand = np.setdiff1d(pool, lab)
                                        if pl in ("tcm_off", "maxherding_off_margin"):
                                            pick = run05.query("margin", logp, cand, q, rng, lab, feats, R)
                                        elif pl == "coreset_off":
                                            pick = op.coreset_greedy(cand, q, lab, hidden)
                                        elif pl in ("typiclust_off_influence", "maxherding_off_influence"):
                                            pick = run05.query("influence", logp, cand, q, rng, lab, feats, R)
                                        elif pl == "age":
                                            pick = op.age_query(logp, cand, q, C, centrality, epoch0=asked, rng=rng, seed=seed)
                                        else:
                                            pick = op.info_density_query(logp, cand, q, feats, unlabeled=cand)
                                        pick = np.asarray(pick)
                                        assert len(np.unique(pick)) == q and np.isin(pick, cand).all()
                                        hits += int((logp.argmax(1)[pick] != labels[pick]).sum())
                                        asked += q
                                        lab = np.sort(np.concatenate([lab, pick]))
                                        if pl == "coreset_off":
                                            logp, hidden = fit(gcn_type, x, ei, labels, lab, seed, **kw)
                                        else:
                                            logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                                assert len(lab) == B and not np.intersect1d(lab, test).size
                                acc = (logp.argmax(1)[test] == labels[test]).mean() * 100
                                rows.append({"key": f"{base}|{b}|{pl}", "dataset": dataset, "descriptor": desc,
                                             "rerank": rr, "gcn_type": gcn_type, "split": sp, "seed": seed,
                                             "budget": b, "n_labeled": B, "pipeline": pl,
                                             "acc": f"{acc:.4f}", "acc_initial": "" if acc0 is None else f"{acc0:.4f}",
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
