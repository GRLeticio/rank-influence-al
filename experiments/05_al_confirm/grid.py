"""
Full grid of experiment 05.

Edit the CONFIGURATION below and run (from the repository root):
  python3 experiments/05_al_confirm/grid.py --dry-run     # lists the jobs, runs nothing
  nohup python3 experiments/05_al_confirm/grid.py > grid05.out 2>&1 &

What happens:
  1. runs the preflight (packages, files, label alignment, memory) -- if it fails, nothing starts
  2. runs the jobs in sequence; each job = 1 call of run.py
     (1 phase x 1 dataset x 1 descriptor x 1 list type)
  3. a failed job is recorded and the grid moves on to the next one
  4. can be killed and restarted: finished jobs are skipped and run.py
     resumes the unfinished ones where they stopped

Monitor:   tail -f grid05.out
           results/grid05/status.csv       (1 line per job: ok / error / time)
           results/grid05/logs/<job>.log   (full output of each job)
Results:   results/05_al_confirm/results.csv
Analysis:  python3 experiments/05_al_confirm/summarize.py
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

# ============================ CONFIGURACAO ============================

# None = usa <repo>/datasets (ou a variavel de ambiente DATASETS_DIR).
DATASETS_DIR = None

DATASETS = ["flowers", "corel5k", "cub200"]
DESCRIPTORS = ["Resnet152", "vitb16", "swintf"]
DESCRIPTORS_BY_DATASET = {}          # ex.: {"cub200": ["Resnet152"]} pra restringir um dataset

# Listas reranqueadas: sufixo do arquivo (Resnet152.txt -> Resnet152_LHRR.txt).
# "none" = listas originais (fase A). O preflight avisa se algum arquivo nao existe.
RERANKS = ["LHRR", "UMAP"]

# Estrategias rodadas nos jobs. None = todas (as do run.py). Desvio de 2026-10-01:
# typiclust, tcm, probcover e probcover_margin saem dos jobs que ainda nao
# rodaram, porque o exp 06 roda as versoes OFICIAIS deles (as nossas nao sao equivalentes).
PIPELINES = ["random", "margin", "coreset", "maxherding", "maxherding_margin",
             "rank_probcover", "rank_probcover_margin", "random_influence", "maxherding_influence", "ours"]

# Fases rodam em ordem: a mais importante primeiro, pra ter resultado util cedo.
PHASES = [
    {   # A: listas originais, as 2 GCNs mais rapidas -- o teste confirmatorio principal
        "name": "A_orig",
        "reranks": ["none"],
        "gcn_types": ["gcn_sgc", "gcn_net"],
    },
    {   # B: listas reranqueadas (muda a entrada: grafo, cobertura e influencia)
        "name": "B_rerank",
        "reranks": RERANKS,
        "gcn_types": ["gcn_sgc", "gcn_net"],
    },
    {   # C: APPNP (a mais lenta), nas duas listas
        "name": "C_appnp",
        "reranks": ["none"] + RERANKS,
        "gcn_types": ["gcn_appnpnet"],
    },
]

# ======================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = os.path.join(ROOT, "experiments", "05_al_confirm", "run.py")
GRID_DIR = os.path.join(ROOT, "results", "grid05")
STATUS = os.path.join(GRID_DIR, "status.csv")
STATUS_FIELDS = ["job", "status", "start", "end", "minutes", "returncode", "command"]


def jobs(phases):
    for ph in phases:
        for ds in DATASETS:                 # do menor pro maior: resultado util cedo
            for desc in DESCRIPTORS_BY_DATASET.get(ds, DESCRIPTORS):
                for rr in ph["reranks"]:
                    name = f"{ph['name']}__{ds}__{desc}__{rr}"
                    cmd = [sys.executable, "-u", RUN, "--datasets", ds, "--descriptors", desc,
                           "--reranks", rr, "--gcn-types", *ph["gcn_types"]]
                    if PIPELINES:
                        cmd += ["--pipelines", *PIPELINES]
                    yield name, cmd


def done_jobs():
    if not os.path.exists(STATUS):
        return set()
    with open(STATUS, newline="") as f:
        return {r["job"] for r in csv.DictReader(f) if r["status"] == "ok"}


def record(row):
    new = not os.path.exists(STATUS)
    with open(STATUS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="so' lista os jobs")
    p.add_argument("--phases", nargs="+", help="rodar so' estas fases (ex.: A_orig)")
    p.add_argument("--skip-preflight", action="store_true")
    a = p.parse_args()

    if DATASETS_DIR:
        os.environ["DATASETS_DIR"] = DATASETS_DIR
    phases = [ph for ph in PHASES if not a.phases or ph["name"] in a.phases]
    all_jobs = list(jobs(phases))
    finished = done_jobs()

    print(f"[{now()}] {len(all_jobs)} jobs ({len(finished & {n for n, _ in all_jobs})} ja' feitos)", flush=True)
    if a.dry_run:
        for name, _ in all_jobs:
            print(("  [feito] " if name in finished else "  ") + name)
        print("\nexemplo de comando:\n  " + " ".join(all_jobs[0][1]))
        return

    if not a.skip_preflight:
        sys.path.insert(0, os.path.join(ROOT, "experiments"))
        import preflight
        reranks = sorted({rr for ph in phases for rr in ph["reranks"]})
        gcns = sorted({g for ph in phases for g in ph["gcn_types"]})
        rc = preflight.main(["--datasets", *DATASETS, "--descriptors", *DESCRIPTORS,
                             "--reranks", *reranks, "--gcn-types", *gcns,
                             "--descriptors-by-dataset", repr(DESCRIPTORS_BY_DATASET)])
        if rc != 0:
            print("Preflight falhou -- grade NAO iniciada.", flush=True)
            sys.exit(1)

    os.makedirs(os.path.join(GRID_DIR, "logs"), exist_ok=True)
    failed = []
    t_all = time.time()
    for i, (name, cmd) in enumerate(all_jobs, 1):
        if name in finished:
            continue
        start = now()
        t0 = time.time()
        print(f"[{start}] ({i}/{len(all_jobs)}) {name}", flush=True)
        with open(os.path.join(GRID_DIR, "logs", f"{name}.log"), "a") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        minutes = (time.time() - t0) / 60
        status = "ok" if rc == 0 else "erro"
        record({"job": name, "status": status, "start": start, "end": now(),
                "minutes": f"{minutes:.1f}", "returncode": rc, "command": " ".join(cmd)})
        print(f"[{now()}]   -> {status} em {minutes:.1f} min", flush=True)
        if rc != 0:
            failed.append(name)

    print(f"\n[{now()}] fim da grade em {(time.time()-t_all)/3600:.1f} h", flush=True)
    if failed:
        print(f"{len(failed)} job(s) com erro (ver results/grid05/logs/):", flush=True)
        for f in failed:
            print("  " + f, flush=True)
        print("Rodar grid.py de novo tenta so' os que faltam.", flush=True)


if __name__ == "__main__":
    main()
