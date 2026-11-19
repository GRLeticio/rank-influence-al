"""
Builds the data package (features, labels and ranked lists) released with the paper.

Uses src/data.py to locate exactly the files read by the experiments, copies the
features and label files, truncates every ranked list to its first L_KEEP positions
(the experiments never read beyond position 100), and then checks that loading the
package with src/data.py gives the same labels, features and lists as the original files.

Usage (on the machine that has the datasets):
  python3 tools/make_data_package.py --datasets-dir /path/to/datasets --out /path/to/output
Output: <out>/rank_influence_al_data_v1.zip, which unpacks to datasets/ (the layout expected
by src/data.py), plus README_DATA.md and MANIFEST.csv (size, shape and sha256 of each file).
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
import zipfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = ["flowers", "corel5k", "cub200"]
DESCRIPTORS = ["Resnet152", "vitb16", "swintf"]
RERANKS = ["", "LHRR", "UMAP"]
L_KEEP = 100
NAME = "rank_influence_al_data_v1"

README_DATA = """# Data for "Rank-Based Influence Queries for Low-Budget Active Learning with GCNs"

Features, labels and ranked lists used in the paper. Code: https://github.com/GRLeticio/rank-influence-al

Unpack this archive in the root of the code repository, so that `datasets/` sits next to `src/`
(or set the environment variable `DATASETS_DIR` to the unpacked `datasets/` folder).

## Contents

Three image collections: Flowers-17, Corel5k and CUB-200-2011. For each, three descriptors
extracted with pre-trained networks: ResNet-152 (2,048-d, global average pooling),
ViT-B/16 (1,000 ImageNet class scores after a sigmoid) and Swin-B (1,024-d, last stage).

- `<dataset>-feat/*.npy` (or `<dataset>/*.npy`): features, one row per image.
- `<dataset>/*.txt`: ranked lists, one line per image, with the indices (0-based) of the most
  similar images by the Euclidean distance between features, the image itself first. Lists
  are truncated to their first {L} positions, the most that the experiments read.
- `*_LHRR.txt`: lists re-ranked with LHRR (UDLF framework); `*_UMAP.txt`: lists re-ranked with
  UMAP, as in Leticio et al., Pattern Recognition Letters 183 (2024). Both are unsupervised.
- `*_classes.txt` (`name:class`, one line per image) or `*_labels.npy`: class labels.
  `corel5k_lists.txt` gives the order of the Corel5k images in the features and lists;
  `src/data.py` reorders the labels accordingly.

`MANIFEST.csv` lists the size, array shape and SHA-256 of every file.

## Images

The images are not included. They can be obtained from the original sources of Flowers-17
(Visual Geometry Group, University of Oxford), Corel5k and CUB-200-2011 (Caltech), under
their own terms of use.

## License

CC BY 4.0. Please cite the paper when using these files.
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_data(datasets_dir):
    """Imports src/data.py with DATASETS_DIR set (data.py reads it at import time)."""
    os.environ["DATASETS_DIR"] = datasets_dir
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import importlib
    import data
    return importlib.reload(data)


def collect(data):
    """(relative path, kind) of every file used by the experiments."""
    files = []
    for ds in DATASETS:
        files.append((os.path.relpath(data.classes_path(ds), data.DATASETS_DIR), "labels"))
        lists = data.lists_path(ds)
        if lists is not None:
            files.append((os.path.relpath(lists, data.DATASETS_DIR), "labels"))
        for desc in DESCRIPTORS:
            files.append((os.path.relpath(data.feat_path(ds, desc), data.DATASETS_DIR), "features"))
            for rr in RERANKS:
                files.append((os.path.relpath(data.rks_path(ds, desc, rr), data.DATASETS_DIR), "ranked_list"))
    missing = [rel for rel, _ in files if not os.path.exists(os.path.join(data.DATASETS_DIR, rel))]
    if missing:
        sys.exit("missing files:\n  " + "\n  ".join(missing))
    return files


def truncate_list(src, dst):
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            fo.write(" ".join(line.split()[:L_KEEP]) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--datasets", nargs="+", default=DATASETS, help="subset, for testing")
    p.add_argument("--reranks", nargs="+", default=["none", "LHRR", "UMAP"], help="subset, for testing")
    a = p.parse_args()
    DATASETS[:] = a.datasets
    RERANKS[:] = ["" if r == "none" else r for r in a.reranks]
    src_dir = os.path.abspath(os.path.expanduser(a.datasets_dir))
    out = os.path.abspath(os.path.expanduser(a.out))
    pkg = os.path.join(out, NAME)
    dst_dir = os.path.join(pkg, "datasets")
    if os.path.exists(pkg):
        sys.exit(f"{pkg} already exists; remove it first")

    data = import_data(src_dir)
    files = collect(data)
    print(f"{len(files)} files from {src_dir}")

    # 1. copy / truncate
    for rel, kind in files:
        s, d = os.path.join(src_dir, rel), os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if kind == "ranked_list":
            truncate_list(s, d)
        else:
            shutil.copy2(s, d)
        print(f"  {kind:12s} {rel}")

    # 2. check: the package loads exactly like the original files
    orig = {}
    for ds in DATASETS:
        orig[ds] = {"labels": data.load_labels(ds)}
        for desc in DESCRIPTORS:
            orig[ds][desc] = data.load_features(ds, desc)
            for rr in RERANKS:
                orig[ds][(desc, rr)] = data.load_rks(ds, desc, L=L_KEEP, rerank=rr)
    data = import_data(dst_dir)
    for ds in DATASETS:
        assert np.array_equal(orig[ds]["labels"], data.load_labels(ds)), ds
        for desc in DESCRIPTORS:
            assert np.array_equal(orig[ds][desc], data.load_features(ds, desc)), (ds, desc)
            for rr in RERANKS:
                assert np.array_equal(orig[ds][(desc, rr)], data.load_rks(ds, desc, L=L_KEEP, rerank=rr)), (ds, desc, rr)
        print(f"check ok: {ds} (n = {len(orig[ds]['labels'])})")

    # 3. manifest and readme
    with open(os.path.join(pkg, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "kind", "bytes", "shape", "sha256"])
        for rel, kind in files:
            d = os.path.join(dst_dir, rel)
            if d.endswith(".npy"):
                shape = "x".join(map(str, np.load(d, mmap_mode="r").shape))
            elif kind == "ranked_list":
                with open(d) as fl:
                    shape = f"{sum(1 for _ in fl)}x{L_KEEP}"
            else:
                shape = ""
            w.writerow([os.path.join("datasets", rel), kind, os.path.getsize(d), shape, sha256(d)])
    with open(os.path.join(pkg, "README_DATA.md"), "w") as f:
        f.write(README_DATA.replace("{L}", str(L_KEEP)))

    # 4. zip
    zpath = os.path.join(out, NAME + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, names in os.walk(pkg):
            for n in sorted(names):
                full = os.path.join(base, n)
                z.write(full, os.path.relpath(full, pkg))
    print(f"\n{zpath}: {os.path.getsize(zpath) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
