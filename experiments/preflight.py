"""
Check BEFORE leaving a grid running: catches in minutes what would make the
server stop in the middle of the night.

Checks:
  1. packages (torch, torch_geometric, sklearn, scipy, pandas), KMeans and GPU
  2. per dataset/descriptor: files exist; features, lists and classes with the
     same n; valid indices; labels ALIGNED with the lists (precision@20,
     catches the Corel5k ordering bug); same for the re-ranked lists
  3. results folder writable and disk space
  4. time/memory on the largest dataset: 1 training per GCN and the expensive
     selections (ProbCover, MaxHerding) at the 10% budget

Usage (from the repository root):
  python3 experiments/preflight.py
  python3 experiments/preflight.py --datasets flowers --descriptors Resnet152 --reranks none LHRR
Exits with code 1 if something fails. grid.py calls it by itself before starting.
"""

import argparse
import ast
import importlib
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data  # noqa: E402

GRAPH_K = 40
L_NEED = 100           # o pacote de dados guarda 100 posicoes; o run.py le max(40, 21)
PREC_K = 20
MIN_PREC = 0.3
MIN_FREE_GB = 2


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def ok(self, msg):
        print(f"  ok    {msg}", flush=True)

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  AVISO {msg}", flush=True)

    def fail(self, msg):
        self.errors.append(msg)
        print(f"  ERRO  {msg}", flush=True)


def check_packages(r):
    print("\n[1] pacotes e GPU")
    for mod in ("torch", "torch_geometric", "sklearn", "scipy", "pandas"):
        try:
            m = importlib.import_module(mod)
            r.ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            r.fail(f"{mod} nao importa: {e}")
    try:
        from sklearn.cluster import KMeans
        KMeans(n_clusters=5, n_init=1, random_state=0).fit(np.random.default_rng(0).normal(size=(200, 8)))
        r.ok("KMeans do sklearn funciona (TypiClust e ProbCover usam)")
    except Exception as e:
        r.fail(f"KMeans quebrou: {type(e).__name__}: {e} -> provavel threadpoolctl antigo: "
               f"pip install -U threadpoolctl")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            r.ok(f"GPU {torch.cuda.get_device_name(0)} ({free/2**30:.1f} de {total/2**30:.1f} GB livres)")
        else:
            r.warn("sem GPU: vai rodar na CPU (bem mais lento)")
    except Exception as e:
        r.fail(f"torch/cuda: {e}")


def _check_lists(r, tag, rks, labels, n, piso, strict):
    problems = []
    if rks.ndim != 2 or rks.shape[0] != n:
        return [f"listas com {len(rks)} linhas, classes tem {n}"]
    if rks.shape[1] < L_NEED:
        problems.append(f"listas com so' {rks.shape[1]} posicoes, precisa >= {L_NEED}")
    if rks.min() < 0 or rks.max() >= n:
        problems.append(f"indices fora de [0, {n-1}]")
    if any(len(set(row)) < len(row) for row in rks[:, :GRAPH_K]):
        problems.append("elemento repetido no top-40 de alguma lista")
    self0 = float((rks[:, 0] == np.arange(n)).mean())
    if self0 < 0.95:
        r.warn(f"{tag}: so' {self0:.0%} das listas comecam pela propria imagem (o codigo assume "
               f"posicao 0 = a propria imagem)")
    if not problems:
        p20 = float((labels[rks[:, 1:PREC_K + 1]] == labels[:, None]).mean())
        if p20 < piso:
            msg = (f"{tag}: precisao@{PREC_K} vs rotulos = {p20:.3f} (esperado > {piso}): rotulos "
                   f"desalinhados com as listas?")
            if strict:
                problems.append(msg)
            else:
                r.warn(msg)
        else:
            r.ok(f"{tag}: listas {rks.shape}, precisao@{PREC_K} vs rotulos = {p20:.3f}")
    return problems


def check_dataset(r, dataset, descriptors, reranks):
    print(f"\n[2] dataset {dataset}  (DATASETS_DIR={data.DATASETS_DIR})")
    if dataset not in data.DATASETS:
        r.fail(f"{dataset} nao esta' em src/data.py")
        return
    try:
        labels = data.load_labels(dataset)
        C = len(np.unique(labels))
        r.ok(f"classes: n={len(labels)}, C={C}, arquivo {data.classes_path(dataset)}")
    except Exception as e:
        r.fail(f"classes de {dataset}: {e}")
        return
    n = len(labels)
    piso = max(MIN_PREC, data.DATASETS[dataset].get("min_p20", 0))
    for desc in descriptors:
        tag = f"{dataset}/{desc}"
        if desc not in data.DATASETS[dataset]["descriptors"]:
            r.fail(f"{tag}: descritor nao configurado em src/data.py")
            continue
        missing = [p for p in data.descriptor_files(dataset, desc) if not os.path.exists(p)]
        if missing:
            for p in missing:
                r.fail(f"{tag}: arquivo nao existe: {p}")
            continue
        try:
            X = data.load_features(dataset, desc)
        except Exception as e:
            r.fail(f"{tag}: erro lendo features: {e}")
            continue
        if X.shape[0] != n:
            r.fail(f"{tag}: features com {X.shape[0]} linhas, classes tem {n}")
            continue
        if not np.isfinite(X).all():
            r.fail(f"{tag}: features com NaN/inf")
        for rr in reranks:
            rerank = "" if rr == "none" else rr
            path = data.rks_path(dataset, desc, rerank)
            ttag = f"{tag} [{rr}]"
            if not os.path.exists(path):
                r.fail(f"{ttag}: arquivo nao existe: {path}")
                continue
            try:
                rks = data.load_rks(dataset, desc, L=L_NEED, rerank=rerank)
            except Exception as e:
                r.fail(f"{ttag}: erro lendo {path}: {e}")
                continue
            # original: desalinhamento e' erro; reranqueada: so' aviso (a precisao pode mudar)
            for p in _check_lists(r, ttag, rks, labels, n, piso, strict=(rr == "none")):
                r.fail(f"{ttag}: {p}")


def check_output(r, out_dir):
    print("\n[3] saida")
    try:
        os.makedirs(out_dir, exist_ok=True)
        probe = os.path.join(out_dir, ".preflight_write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        r.ok(f"gravavel: {out_dir}")
    except Exception as e:
        r.fail(f"nao consigo gravar em {out_dir}: {e}")
        return
    free = shutil.disk_usage(out_dir).free / 2**30
    (r.ok if free >= MIN_FREE_GB else r.fail)(f"espaco livre: {free:.1f} GB")


def check_memory(r, datasets, descriptors, gcn_types):
    print("\n[4] tempo e memoria no maior dataset")
    import gcn as _gcn
    print(f"  (formato do grafo: {_gcn.ADJ_FORMAT}; mude com GCN_ADJ=csr se faltar memoria de GPU)")
    import torch
    import coverage as cv
    from gcn import fit
    from graphs import build_edge_index
    from protocol import fixed_test_split
    sizes = [(len(data.load_labels(ds)), ds) for ds in datasets]
    n, ds = max(sizes)
    desc = max((d for d in descriptors if os.path.exists(data.feat_path(ds, d))),
               key=lambda d: np.load(data.feat_path(ds, d), mmap_mode="r").shape[1], default=None)
    if desc is None:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = data.load_features(ds, desc)
    labels = data.load_labels(ds)
    rks = data.load_rks(ds, desc, L=L_NEED)
    x = torch.from_numpy(X).to(device)
    ei = build_edge_index(rks, GRAPH_K, "rec").to(device)
    train = np.arange(0, n, 10)
    for g in gcn_types:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            fit(g, x, ei, labels, train, 0, **data.gcn_params(ds))
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            r.ok(f"{ds}/{desc} {g}: {time.time()-t:.1f}s por treino, pico de GPU {peak:.2f} GB")
        except Exception as e:
            msg = str(e).lower()
            oom = ("out of memory" in msg or "alloc_failed" in msg or type(e).__name__ == "OutOfMemoryError")
            if not oom:
                r.fail(f"{ds}/{desc} {g}: {type(e).__name__}: {e}")
                continue
            torch.cuda.empty_cache()
            t = time.time()
            fit(g, x.cpu(), ei.cpu(), labels, train, 0, **data.gcn_params(ds))
            r.warn(f"{ds}/{desc} {g}: SEM MEMORIA de GPU -> o run.py treina na CPU nesses casos "
                   f"({time.time()-t:.1f}s por treino na CPU). Alternativa: GCN_ADJ=csr, se o torch "
                   f"do servidor aceitar (testar com: GCN_ADJ=csr python3 experiments/preflight.py ...)")
    pool, _ = fixed_test_split(labels, 0)
    m = int(round(0.10 * n))
    try:
        t = time.time()
        cv.select_probcover(pool, m, X, int(labels.max()) + 1, tag="preflight")
        r.ok(f"ProbCover em {ds} (m={m}): {time.time()-t:.1f}s")
        t = time.time()
        cv.select_maxherding(pool, m, X, tag="preflight")
        r.ok(f"MaxHerding em {ds} (m={m}): {time.time()-t:.1f}s")
    except Exception as e:
        r.fail(f"selecoes de cobertura em {ds}: {type(e).__name__}: {e}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["flowers", "corel5k", "cub200"])
    p.add_argument("--descriptors", nargs="+", default=["Resnet152", "vitb16", "swintf"])
    p.add_argument("--descriptors-by-dataset", default="{}", help="dict em texto, como no grid.py")
    p.add_argument("--reranks", nargs="+", default=["none"])
    p.add_argument("--gcn-types", nargs="+", default=["gcn_sgc", "gcn_net", "gcn_appnpnet"])
    p.add_argument("--out", default=os.path.join(ROOT, "results"))
    p.add_argument("--skip-memory", action="store_true")
    a = p.parse_args(argv)
    by_ds = ast.literal_eval(a.descriptors_by_dataset)

    r = Report()
    check_packages(r)
    for ds in a.datasets:
        check_dataset(r, ds, by_ds.get(ds, a.descriptors), a.reranks)
    check_output(r, a.out)
    if not r.errors and not a.skip_memory:
        try:
            check_memory(r, a.datasets, a.descriptors, a.gcn_types)
        except Exception as e:
            r.fail(f"checagem de memoria: {type(e).__name__}: {e}")

    print(f"\nResumo: {len(r.errors)} erro(s), {len(r.warnings)} aviso(s)")
    for e in r.errors:
        print(f"  ERRO  {e}")
    return 1 if r.errors else 0


if __name__ == "__main__":
    sys.exit(main())
