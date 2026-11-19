"""
Grid of experiment 09 (budget curves). Reuses run.py of exp 06 WITHOUT any change:
it only passes other budgets (--budgets) and another output folder (--out).

  python3 experiments/09_budget_curves/grid.py --dry-run
  nohup python3 experiments/09_budget_curves/grid.py > grid09.out 2>&1 &

Monitor: tail -f grid09.out ; results/grid09/status.csv ; results/09_budget_curves/log.txt
Can be killed and restarted: finished jobs are skipped and run.py resumes where it stopped.
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time

# ============================ CONFIGURACAO ============================
DATASETS_DIR = os.environ.get("DATASETS_DIR")      # None = <repo>/datasets

DATASETS = ["corel5k", "cub200"]
DESCRIPTORS = ["Resnet152", "vitb16", "swintf"]
GCN_TYPES = ["gcn_sgc", "gcn_net"]
BUDGETS = ["0.01", "0.03", "0.07"]          # 2/5/10% ja' estao no exp 06
PIPELINES = ["maxherding_off", "maxherding_off_margin", "maxherding_off_influence",
             "typiclust_off", "tcm_off", "typiclust_off_influence"]
# ======================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = os.path.join(ROOT, "experiments", "06_al_baselines", "run.py")
OUT = os.path.join(ROOT, "results", "09_budget_curves")
OUT_CHECK = os.path.join(ROOT, "results", "09_budget_curves_check")
GRID_DIR = os.path.join(ROOT, "results", "grid09")
STATUS = os.path.join(GRID_DIR, "status.csv")
STATUS_FIELDS = ["job", "status", "start", "end", "minutes", "returncode", "command"]


def jobs():
    # conferencia: 5% nas 10 divisoes de uma configuracao; a MEDIA tem que bater com o exp 06
    # (o treino da GCN na GPU nao e' bit a bit deterministico, entao nao da' pra exigir igualdade exata)
    yield "check__corel5k__vitb16", [sys.executable, "-u", RUN, "--datasets", "corel5k", "--descriptors", "vitb16",
                                     "--reranks", "none", "--gcn-types", "gcn_net", "--budgets", "0.05",
                                     "--splits", "10", "--pipelines", *PIPELINES, "--out", OUT_CHECK]
    for ds in DATASETS:
        for desc in DESCRIPTORS:
            yield f"curves__{ds}__{desc}", [sys.executable, "-u", RUN, "--datasets", ds, "--descriptors", desc,
                                            "--reranks", "none", "--gcn-types", *GCN_TYPES,
                                            "--budgets", *BUDGETS, "--pipelines", *PIPELINES, "--out", OUT]


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
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if DATASETS_DIR:
        os.environ["DATASETS_DIR"] = DATASETS_DIR
    print(f"[{now()}] DATASETS_DIR={os.environ.get('DATASETS_DIR', '(padrao do data.py)')}", flush=True)
    all_jobs = list(jobs())
    finished = done_jobs()
    if a.dry_run:
        for name, cmd in all_jobs:
            print(("  [feito] " if name in finished else "  ") + name)
        print("\nexemplo:\n  " + " ".join(all_jobs[1][1]))
        return
    os.makedirs(os.path.join(GRID_DIR, "logs"), exist_ok=True)
    t_all = time.time()
    failed = []
    for i, (name, cmd) in enumerate(all_jobs, 1):
        if name in finished:
            continue
        start, t0 = now(), time.time()
        print(f"[{start}] ({i}/{len(all_jobs)}) {name}", flush=True)
        with open(os.path.join(GRID_DIR, "logs", f"{name}.log"), "a") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        minutes = (time.time() - t0) / 60
        status = "ok" if rc == 0 else "erro"
        record({"job": name, "status": status, "start": start, "end": now(), "minutes": f"{minutes:.1f}",
                "returncode": rc, "command": " ".join(cmd)})
        print(f"[{now()}]   -> {status} em {minutes:.1f} min", flush=True)
        if rc != 0:
            failed.append(name)
    print(f"\n[{now()}] fim da grade em {(time.time()-t_all)/3600:.1f} h", flush=True)
    for f in failed:
        print("  ERRO: " + f, flush=True)


if __name__ == "__main__":
    main()
