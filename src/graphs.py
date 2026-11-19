"""
Graph construction from the ranked lists.

"knn" and "rec" produce the SAME edge set as the original code of our group
(GCNClassifier.create_graph), only vectorized.

About direction: the edge is [i, j] with j in the top-k of i. In PyG messages
go from edge_index[0] to edge_index[1], so in "knn" each node aggregates from
the nodes that have it in their top-k (reverse neighbors), not from its own
neighbors. "rec" is symmetric and has no such effect. "knn_und" symmetrizes
knn (union of both directions) to measure the impact of this.
"""

import numpy as np
import torch
from torch_geometric.utils import to_undirected


def build_edge_index(rks, top_k=40, graph_type="rec"):
    rks = np.asarray(rks)[:, :top_k]
    n = rks.shape[0]
    src = np.repeat(np.arange(n), top_k)
    dst = rks.reshape(-1)

    if graph_type in ("knn", "knn_und"):
        edge_index = torch.from_numpy(np.stack([src, dst])).long()
        if graph_type == "knn_und":
            edge_index = to_undirected(edge_index, num_nodes=n)
        return edge_index

    if graph_type == "rec":
        # aresta (i, j) fica se i tambem esta' no top-k de j
        member = set(zip(src.tolist(), dst.tolist()))
        keep = np.fromiter(((d, s) in member for s, d in zip(src.tolist(), dst.tolist())),
                           dtype=bool, count=len(src))
        return torch.from_numpy(np.stack([src[keep], dst[keep]])).long()

    raise ValueError(f"graph_type desconhecido: {graph_type}")
