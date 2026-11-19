"""
Grid of experiment 10 (TCM with rule 1 of its authors).

  python3 experiments/10_tcm_rule/grid.py --dry-run
  nohup python3 experiments/10_tcm_rule/grid.py > grid10.out 2>&1 &

Can be killed and restarted: finished jobs are skipped and run.py resumes where it stopped.
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

DATASETS_DIR = os.environ.get("DATASETS_DIR")      # None = <repo>/datasets
DATASETS = ["flowers", "corel5k", "cub200"]
DESCRIPTORS = ["Resnet152", "vitb16", "swintf"]
PHASES = [("A_orig", ["none"]), ("B_rerank", ["LHRR", "UMAP"])]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = os.path.join(ROOT, "experiments", "10_tcm_rule", "run.py")
GRID_DIR = os.path.join(ROOT, "results", "grid10")
STATUS = os.path.join(GRID_DIR, "status.csv")
FIELDS = ["job", "status", "start", "end", "minutes", "returncode", "command"]


def jobs(query, phases):
    tag = "" if query == "margin" else "INFL_"
    for ph, reranks in PHASES:
        if phases and ph not in phases:
            continue
        for ds in DATASETS:
            for desc in DESCRIPTORS:
                for rr in reranks:
                    yield f"{tag}{ph}__{ds}__{desc}__{rr}", [sys.executable, "-u", RUN, "--datasets", ds,
                                                              "--descriptors", desc, "--reranks", rr,
                                                              "--query", query]


def done_jobs():
    if not os.path.exists(STATUS):
        return set()
    with open(STATUS, newline="") as f:
        return {r["job"] for r in csv.DictReader(f) if r["status"] == "ok"}


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--query", choices=["margin", "influence"], default="margin",
                   help="margin = TCM pela regra dos autores; influence = mesmo esquema com a nossa influencia")
    p.add_argument("--phases", nargs="+", help="ex.: A_orig (so' listas originais)")
    a = p.parse_args()
    if DATASETS_DIR:
        os.environ["DATASETS_DIR"] = DATASETS_DIR
    print(f"[{now()}] DATASETS_DIR={os.environ.get('DATASETS_DIR', '(padrao do data.py)')}", flush=True)
    all_jobs, finished = list(jobs(a.query, a.phases)), done_jobs()
    if a.dry_run:
        for name, _ in all_jobs:
            print(("  [feito] " if name in finished else "  ") + name)
        return
    os.makedirs(os.path.join(GRID_DIR, "logs"), exist_ok=True)
    t_all, failed = time.time(), []
    for i, (name, cmd) in enumerate(all_jobs, 1):
        if name in finished:
            continue
        start, t0 = now(), time.time()
        print(f"[{start}] ({i}/{len(all_jobs)}) {name}", flush=True)
        with open(os.path.join(GRID_DIR, "logs", f"{name}.log"), "a") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        status = "ok" if rc == 0 else "erro"
        new = not os.path.exists(STATUS)
        with open(STATUS, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({"job": name, "status": status, "start": start, "end": now(),
                        "minutes": f"{(time.time()-t0)/60:.1f}", "returncode": rc, "command": " ".join(cmd)})
        print(f"[{now()}]   -> {status} em {(time.time()-t0)/60:.1f} min", flush=True)
        if rc != 0:
            failed.append(name)
    print(f"\n[{now()}] fim da grade em {(time.time()-t_all)/3600:.1f} h", flush=True)
    for f in failed:
        print("  ERRO: " + f, flush=True)


if __name__ == "__main__":
    main()
