"""
Experiment 05 -- confirmatory experiment of rank-based active learning.

Frozen method ("ours"): Rank-ProbCover for the first B/2 labels + greedy rank-based
influence for the remaining B/2 (5 rounds). Compared with baselines and ablations
at budgets of 2/5/10% of n, fixed 20% test set, 10 splits.

One "unit" = dataset x descriptor x list (original/re-ranked) x GCN x split x seed.
For each unit and budget, all pipelines are run and each writes 1 line to
results.csv. Resumes where it stopped (keys already written are skipped).

Usage (from the repository root):
  python3 experiments/05_al_confirm/run.py --smoke
  python3 experiments/05_al_confirm/run.py --datasets flowers --descriptors Resnet152 --reranks none
  (grid.py calls this with 1 dataset x 1 descriptor x 1 list type at a time)
"""

import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import coverage as cv
import data
import selection
from active import (initial_rank_probcover, pick_coreset, pick_margin, pick_rank_influence_greedy,
                    reverse_matrix, uncertainty_prob)
from gcn import fit
from graphs import build_edge_index
from protocol import fixed_test_split

# ---- congelado ----
GRAPH_K = 40
COVER_K = 10
INFL_K, INFL_P = 20, 0.9
ROUNDS = 5
BUDGETS = (0.02, 0.05, 0.10)

# nome: (selecao inicial, consulta). consulta None = a selecao inicial pega o orcamento todo
PIPELINES = {
    "random": ("random", "random"),
    "margin": ("random", "margin"),
    "coreset": ("random", "coreset"),
    "tcm": ("typiclust", "margin"),
    "typiclust": ("typiclust", None),
    "probcover": ("probcover", None),
    "maxherding": ("maxherding", None),
    "probcover_margin": ("probcover", "margin"),
    "maxherding_margin": ("maxherding", "margin"),
    # ablacoes
    "rank_probcover": ("rank_probcover", None),
    "rank_probcover_margin": ("rank_probcover", "margin"),
    "random_influence": ("random", "influence"),
    "maxherding_influence": ("maxherding", "influence"),
    # metodo principal
    "ours": ("rank_probcover", "influence"),
}
# selecoes gulosas: a de tamanho m e' o prefixo da de tamanho M > m -> calcula 1x no maior
GREEDY_INITIALS = {"rank_probcover", "probcover", "maxherding"}

FIELDS = ["key", "dataset", "descriptor", "rerank", "gcn_type", "split", "seed", "budget", "n_labeled",
          "pipeline", "acc", "acc_initial", "hit_rate", "coverage", "seconds"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["flowers", "corel5k", "cub200"])
    p.add_argument("--descriptors", nargs="+", default=["Resnet152", "vitb16", "swintf"])
    p.add_argument("--reranks", nargs="+", default=["none"],
                   help='"none" = listas originais; ex.: LHRR -> arquivos <lista>_LHRR.txt')
    p.add_argument("--gcn-types", nargs="+", default=["gcn_sgc", "gcn_net", "gcn_appnpnet"])
    p.add_argument("--pipelines", nargs="+", default=list(PIPELINES))
    p.add_argument("--budgets", nargs="+", type=float, default=list(BUDGETS))
    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "05_al_confirm"))
    p.add_argument("--smoke", action="store_true", help="1 divisao, Flowers/ResNet/SGC, orcamento 10%%")
    a = p.parse_args()
    if a.smoke:
        a.datasets, a.descriptors, a.gcn_types = ["flowers"], ["Resnet152"], ["gcn_sgc"]
        a.splits, a.budgets = 1, [0.10]
        a.out = os.path.join(ROOT, "results", "05_al_confirm_smoke")
    return a


def log(msg, path):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def completed_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["key"] for row in csv.DictReader(f)}


def append_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


class Initials:
    """Selecoes iniciais (sem modelo) de uma divisao, com cache. Nenhuma usa rotulo do pool."""

    def __init__(self, pool, feats, rks, n_classes, tag, max_m):
        self.pool, self.feats, self.rks = pool, feats, rks
        self.n_classes, self.tag, self.max_m = n_classes, tag, max_m
        self.cache = {}

    def get(self, kind, m, seed):
        if kind == "random":
            return np.sort(np.random.default_rng(10_000 + seed).permutation(self.pool)[:m])
        key = (kind, None if kind in GREEDY_INITIALS else m, seed)
        if key not in self.cache:
            M = self.max_m if kind in GREEDY_INITIALS else m
            if kind == "rank_probcover":
                sel = _ordered_rank_probcover(self.pool, M, self.rks)
            elif kind == "probcover":
                sel = cv.select_probcover(self.pool, M, self.feats, self.n_classes, seed=seed, tag=self.tag,
                                          ordered=True)[0]
            elif kind == "maxherding":
                sel = cv.select_maxherding(self.pool, M, self.feats, tag=self.tag, ordered=True)[0]
            elif kind == "typiclust":
                sel = np.asarray(selection.select_typiclust(self.pool, m, seed, self.feats, tag=self.tag))
            else:
                raise ValueError(kind)
            self.cache[key] = sel
        sel = self.cache[key]
        out = np.sort(sel[:m])
        assert len(out) == m and len(np.unique(out)) == m
        return out


def _ordered_rank_probcover(pool, m, rks):
    """initial_rank_probcover devolve ordenado por indice; aqui precisamos da ORDEM de escolha
    (pro prefixo). Mesmo algoritmo, guardando a ordem."""
    rks = np.asarray(rks)
    n = rks.shape[0]
    balls = rks[:, :COVER_K + 1]
    covered = np.zeros(n, dtype=bool)
    avail = np.zeros(n, dtype=bool)
    avail[pool] = True
    noise = np.random.default_rng(0).random(n) * 1e-6
    out = []
    for _ in range(m):
        gain = (~covered[balls]).sum(axis=1).astype(float)
        v = int(np.argmax(np.where(avail, gain + noise, -np.inf)))
        out.append(v)
        avail[v] = False
        covered[balls[v]] = True
    return np.asarray(out)


def _is_oom(e):
    """Falta de memoria de GPU aparece com mensagens diferentes conforme onde estoura
    (conferido: no SGC/CUB veio como CUBLAS_STATUS_ALLOC_FAILED, sem "out of memory")."""
    msg = str(e).lower()
    return (isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in msg
            or "alloc_failed" in msg or "cudnn_status_alloc" in msg)


class SafeFit:
    """
    fit() com protecao contra falta de memoria de GPU:
      1. se der OOM: libera o cache, espera (a GPU pode estar sendo usada por outra
         pessoa) e tenta de novo, ate' OOM_RETRIES vezes;
      2. se continuar: treina na CPU (mais lento, mesmo resultado a menos de
         arredondamento) e registra no log. Daí em diante, naquele job, essa GCN vai
         direto pra CPU (sem esperar de novo a cada treino).
    Medido no CUB/ResNet152 (2048-d) com edge_index: SGC passa de 4 GB (propaga as
    features cruas por 191 mil arestas); GCN-Net e APPNP ~0,5 GB.
    """
    OOM_RETRIES = 2
    WAIT_S = int(os.environ.get("AL_OOM_WAIT", 60))

    def __init__(self, x, ei, log_path):
        self.x, self.ei, self.log_path = x, ei, log_path
        self.cpu = None
        self.cpu_only = set()                       # GCNs que ja' cairam pra CPU neste job
        self.n_cpu = 0

    def __call__(self, gcn_type, labels, lab, seed, **kw):
        if self.x.device.type == "cuda" and gcn_type not in self.cpu_only:
            for attempt in range(self.OOM_RETRIES + 1):
                try:
                    return fit(gcn_type, self.x, self.ei, labels, lab, seed, **kw)[0]
                except (RuntimeError, MemoryError) as e:
                    if not _is_oom(e):
                        raise
                    torch.cuda.empty_cache()
                    if attempt < self.OOM_RETRIES:
                        log(f"  sem memoria de GPU ({gcn_type}); nova tentativa em {self.WAIT_S}s", self.log_path)
                        time.sleep(self.WAIT_S)
            self.cpu_only.add(gcn_type)
            if self.cpu is None:
                self.cpu = (self.x.cpu(), self.ei.cpu())
        if gcn_type in self.cpu_only:
            self.n_cpu += 1
            if self.n_cpu == 1 or self.n_cpu % 100 == 0:
                log(f"  {gcn_type} sem memoria de GPU: treinando na CPU ({self.n_cpu} treinos ate' agora)",
                    self.log_path)
            return fit(gcn_type, *self.cpu, labels, lab, seed, **kw)[0]
        return fit(gcn_type, self.x, self.ei, labels, lab, seed, **kw)[0]


def query(kind, logp, cand, q, rng, lab, feats, R):
    if kind == "random":
        return rng.choice(cand, q, replace=False)
    if kind == "margin":
        # margem classica 1 - (p1 - p2); ordena igual em praticamente todos os casos a do PL21
        return cand[np.lexsort((rng.random(len(cand)), -uncertainty_prob(logp)[cand]))[:q]]
    if kind == "coreset":
        return pick_coreset(cand=cand, q=q, labeled=lab, feats=feats)
    if kind == "influence":
        return pick_rank_influence_greedy(logp=logp, cand=cand, q=q, rng=rng, labeled=lab, R=R,
                                          unc_fn=uncertainty_prob)
    raise ValueError(kind)


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "results.csv")
    log_path = os.path.join(a.out, "log.txt")
    done = completed_keys(csv_path)
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
            for rr in a.reranks:
                rerank = "" if rr == "none" else rr
                rks = data.load_rks(dataset, desc, L=max(GRAPH_K, INFL_K + 1), rerank=rerank)
                ei = build_edge_index(rks, GRAPH_K, "rec").to(device)
                R = reverse_matrix(rks, INFL_K, INFL_P)
                safe_fit = SafeFit(x, ei, log_path)
                tag_base = f"{dataset}|{desc}|{rr}"
                for sp in range(a.splits):
                    pool, test = fixed_test_split(labels, sp)
                    selection.clear_cache()
                    inits = Initials(pool, feats, rks, C, tag=f"{dataset}|{desc}|split{sp}",
                                     max_m=max(budgets.values()))
                    for gcn_type in a.gcn_types:
                        for seed in range(a.seeds):
                            base = f"{tag_base}|{gcn_type}|{sp}|{seed}"
                            todo = [(b, pl) for b in budgets for pl in a.pipelines
                                    if f"{base}|{b}|{pl}" not in done]
                            if not todo:
                                continue
                            t_unit = time.time()
                            rows = []
                            fitted = {}                      # (inicial, m) -> logp do modelo inicial
                            for b, pl in todo:
                                B = budgets[b]
                                init_kind, q_kind = PIPELINES[pl]
                                m0 = B if q_kind is None else B // 2
                                t0 = time.time()
                                lab = inits.get(init_kind, m0, seed)
                                assert not np.intersect1d(lab, test).size
                                if (init_kind, m0) not in fitted:
                                    fitted[(init_kind, m0)] = safe_fit(gcn_type, labels, lab, seed, **kw)
                                logp = fitted[(init_kind, m0)]
                                acc0 = (logp.argmax(1)[test] == labels[test]).mean() * 100
                                hits = asked = 0
                                if q_kind is not None:
                                    rng = np.random.default_rng(20_000 + 100 * sp + seed)
                                    for q in [len(c) for c in np.array_split(np.arange(B - m0), ROUNDS)]:
                                        if q == 0:
                                            continue
                                        cand = np.setdiff1d(pool, lab)
                                        pick = np.asarray(query(q_kind, logp, cand, q, rng, lab, feats, R))
                                        assert len(np.unique(pick)) == q and np.isin(pick, cand).all()
                                        hits += int((logp.argmax(1)[pick] != labels[pick]).sum())
                                        asked += q
                                        lab = np.sort(np.concatenate([lab, pick]))
                                        logp = safe_fit(gcn_type, labels, lab, seed, **kw)
                                assert len(lab) == B and not np.intersect1d(lab, test).size
                                acc = (logp.argmax(1)[test] == labels[test]).mean() * 100
                                rows.append({"key": f"{base}|{b}|{pl}", "dataset": dataset, "descriptor": desc,
                                             "rerank": rr, "gcn_type": gcn_type, "split": sp, "seed": seed,
                                             "budget": b, "n_labeled": B, "pipeline": pl,
                                             "acc": f"{acc:.4f}", "acc_initial": f"{acc0:.4f}",
                                             "hit_rate": f"{hits/asked:.4f}" if asked else "",
                                             "coverage": f"{len(np.unique(labels[lab]))/C:.4f}",
                                             "seconds": f"{time.time()-t0:.1f}"})
                            append_rows(csv_path, rows)
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
