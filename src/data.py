"""
Dataset loading: labels, features and ranked lists.

Datasets are expected in <repo>/datasets (see README). To use another folder, set
the environment variable DATASETS_DIR.

Re-ranked lists: load_rks(..., rerank="LHRR") reads <file>_LHRR.txt.

Ranked list <-> feature pairs checked on 2026-09-29: the lists are the Euclidean
kNN of the corresponding feature (overlap@10 = 1.0 for all pairs below).
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.environ.get(
    "DATASETS_DIR",
    os.path.normpath(os.path.join(_HERE, "..", "datasets")),
)

# gcn: hiperparametros da GCN por dataset (opcional; vazio = defaults de
# src/gcn.py). Hoje nenhum dataset sobrescreve nada. O texto do DGCG diz
# lr 1e-2 no CUB, mas a Tabela I dele so' reproduz com 1e-3 (conferido em
# 2026-09-30: APPNP/REC/ResNet 48,66 com 1e-3 e 42,69 com 1e-2; artigo 48,39).
DATASETS = {
    "flowers": {
        "classes": ["flowers-feat/flowers_classes.txt", "flowers/flowers_classes.txt"],
        "min_p20": 0.6,     # preflight; certo 0.73-1.00
        "descriptors": {
            "Resnet152": [("flowers/Resnet152.txt", "flowers-feat/Resnet152.npy")],
            "vitb16": [("flowers/vitb16.txt", "flowers-feat/vitb16.npy"),
                       ("flowers/VIT-B16.txt", "flowers-feat/VIT-B16.npy")],
            "swintf": [("flowers/swintf_flowers.txt", "flowers-feat/swintf.npy"),
                       ("flowers/SWIN-TF.txt", "flowers-feat/SWIN-TF.npy")],
            # no servidor (sibgrapi2021_experiments/datasets) nao ha' DINOv2 do Flowers
            "dinov2": [("flowers/dinov2_vitb14.txt", "flowers-feat/dinov2_vitb14.npy")],
        },
    },
    "corel5k": {
        # classes.txt em ordem numerica; load_labels reordena pelo _lists.txt.
        # Conferido em 2026-09-30: resultado identico a corel5k_class de
        # sibgrapi2021_experiments/corel5k_evaluation.py (servidor)
        "classes": ["corel5k-feat/corel5k_classes.txt", "corel5k/corel5k_classes.txt"],
        "min_p20": 0.75,    # preflight; certo 0.85-0.93, desalinhado (bug) 0.51-0.56
        "descriptors": {
            "Resnet152": [("corel5k/Resnet152.txt", "corel5k-feat/Resnet152.npy")],
            "vitb16": [("corel5k/VIT-B16.txt", "corel5k-feat/VIT-B16.npy")],
            "swintf": [("corel5k/swimtf.txt", "corel5k-feat/swimtf.npy"),
                       ("corel5k/SWIN-TF.txt", "corel5k-feat/SWIN-TF.npy")],
            "dinov2": [("corel5k/dinov2_vitb14.txt", "corel5k-feat/dinov2_vitb14.npy"),
                       ("corel5k/DINOv2.txt", "corel5k-feat/DINOv2.npy")],
        },
    },
    "cub200": {
        # no servidor o CUB tem so' o .npy (sibgrapi2021_experiments/cub_200_evaluation.py);
        # conferido em 2026-09-30: mesmos rotulos do .txt
        "classes": ["cub200-feat/cub200_classes.txt", "cub200/cub200_classes.txt",
                    "cub200/cub200_labels.npy", "cub200-feat/cub200_labels.npy"],
        "min_p20": 0.3,     # preflight; certo 0.36 (ResNet) a 0.76
        "descriptors": {
            "Resnet152": [("cub200/resnet152.txt", "cub200-feat/resnet152.npy")],
            "vitb16": [("cub200/vit-b16.txt", "cub200-feat/vit-b16.npy")],
            "swintf": [("cub200/swimtf.txt", "cub200-feat/swimtf.npy")],
            "dinov2": [("cub200/dinov2_vitb14.txt", "cub200-feat/dinov2_vitb14.npy")],
        },
    },
}


def gcn_params(dataset):
    return DATASETS[dataset].get("gcn", {})


def _path(rel):
    return os.path.join(DATASETS_DIR, rel)


def classes_path(dataset):
    """Primeiro candidato que existe (local fica em *-feat/, no servidor em <dataset>/)."""
    cands = [_path(c) for c in DATASETS[dataset]["classes"]]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"arquivo de classes de {dataset} nao encontrado em: {cands}")


def lists_path(dataset):
    """<dataset>_lists.txt (ordem das features/ranked lists), procurado nas pastas
    de todos os candidatos de classes; None se nao existir (CUB)."""
    for c in DATASETS[dataset]["classes"]:
        cand = os.path.join(os.path.dirname(_path(c)), f"{dataset}_lists.txt")
        if os.path.exists(cand):
            return cand
    return None


def load_labels(dataset):
    path = classes_path(dataset)
    if path.endswith(".npy"):
        labels = np.load(path).tolist()
    else:
        with open(path) as f:
            pares = [linha.strip().rsplit(":", 1) for linha in f if linha.strip()]
        labels = [int(c) for _, c in pares]
        # BUG corrigido em 2026-09-30: no Corel5k o <dataset>_classes.txt esta'
        # em ordem numerica (1.jpg, 2.jpg...) e as features/ranked lists na
        # ordem do <dataset>_lists.txt, lexicografica (1.jpg, 10.jpg, 100.jpg...).
        # Sem reordenar, os rotulos ficam embaralhados (GCN ~55% em vez de ~95%).
        # No Flowers as duas ordens coincidem. O preflight confere o alinhamento
        # (precisao@20 das ranked lists contra os rotulos).
        lists = lists_path(dataset)
        if lists is not None:
            por_nome = {nome: int(c) for nome, c in pares}
            with open(lists) as f:
                labels = [por_nome[linha.strip()] for linha in f if linha.strip()]
    # remapeia pra 0..C-1 (CUB vem 1..200); identidade em flowers/corel5k
    _, labels = np.unique(np.asarray(labels), return_inverse=True)
    return labels.astype(np.int64)


def descriptor_files(dataset, descriptor):
    """
    (ranked lists, features) do descritor: o primeiro par candidato em que os
    DOIS arquivos existem (nomes mudam entre esta maquina e o servidor). Se
    nenhum existe, devolve o primeiro, pra mensagem de erro mostrar o caminho.
    """
    cands = [(_path(rk), _path(ft)) for rk, ft in DATASETS[dataset]["descriptors"][descriptor]]
    for rk, ft in cands:
        if os.path.exists(rk) and os.path.exists(ft):
            return rk, ft
    return cands[0]


def load_features(dataset, descriptor):
    return np.asarray(np.load(feat_path(dataset, descriptor)), dtype=np.float32)


def load_rks(dataset, descriptor, L=100, rerank=""):
    """Ranked lists truncadas em L, como matriz int (n, L).
    rerank: "" = listas originais; "LHRR", "UMAP"... = arquivo reranqueado,
    mesmo nome com sufixo _<METODO> (ex.: Resnet152.txt -> Resnet152_LHRR.txt)."""
    with open(rks_path(dataset, descriptor, rerank)) as f:
        rks = [[int(y) for y in x.split()[:L]] for x in f]
    return np.asarray(rks, dtype=np.int64)


def feat_path(dataset, descriptor):
    return descriptor_files(dataset, descriptor)[1]


def rks_path(dataset, descriptor, rerank=""):
    base = descriptor_files(dataset, descriptor)[0]
    if not rerank:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}_{rerank}{ext}"
