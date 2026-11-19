"""
GCN models and training (from the code of our group's previous rank-based GCN papers).

Changes with respect to the original, without changing the baseline:
  - each model stores self.hidden (representation before the output layer);
  - fit() returns the log-probabilities and hidden representations of all nodes and
    accepts weighted pseudo-labels (idx, class, weight) in the loss.
  train_and_predict() is unchanged (checked: same predictions).

Architectures and hyperparameters as in the previous papers of the group, to keep
the same baseline: Adam, lr=10e-4, weight_decay=5e-4, 200 epochs, 256 hidden
units, evaluation at the last epoch, no validation set.

10e-4 = 1e-3, as in DGCG (SIBGRAPI 2025); some of the earlier texts say 10^-4
(a typo). Reproduction of Table I of DGCG checked with this code:
Flowers/ResNet152/SGC/REC = 83.96 (paper: 83.98).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SGConv, APPNP

# Formato do grafo passado pras camadas:
#   "edge_index" (default): igual ao codigo do grupo. Materializa 1 copia da
#       feature por aresta -- o SGC em Corel5k/knn pede ~2 GB de GPU.
#   "csr": esparso nativo do torch, sem essa copia (pra GPU pequena, ex.: a
#       RTX 3050 de 4 GB). Precisa de torch recente: no servidor (torch
#       antigo, py3.8) deu "CUDA error: invalid argument".
# Os dois dao a mesma saida (conferido: diferenca max ~1e-6). Escolha com a
# variavel de ambiente GCN_ADJ=csr.
ADJ_FORMAT = os.environ.get("GCN_ADJ", "edge_index")

LR = 10e-4
WEIGHT_DECAY = 5e-4
EPOCHS = 200
HIDDEN = 256


class Net(torch.nn.Module):
    def __init__(self, n_feat, n_hidden, n_classes):
        super().__init__()
        self.conv1 = GCNConv(n_feat, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        self.hidden = x
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    def __init__(self, n_feat, n_classes):
        super().__init__()
        self.conv1 = SGConv(n_feat, n_classes, K=2, cached=True)

    def forward(self, x, edge_index, edge_weight=None):
        # sem camada escondida: o "hidden" e' a propria saida (logits)
        self.hidden = self.conv1(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(self.hidden, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self, n_feat, n_classes, hidden=64, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = Linear(n_feat, hidden)
        self.lin2 = Linear(hidden, n_classes)
        self.prop1 = APPNP(K, alpha)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        self.hidden = x
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.prop1(self.lin2(x), edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


def make_model(gcn_type, n_feat, n_classes):
    if gcn_type == "gcn_net":
        return Net(n_feat, HIDDEN, n_classes)
    if gcn_type == "gcn_sgc":
        return SGC(n_feat, n_classes)
    if gcn_type == "gcn_appnpnet":
        return APPNPNet(n_feat, n_classes)
    raise ValueError(f"gcn_type desconhecido: {gcn_type}")


def _graph_arg(x, edge_index, edge_weight):
    if ADJ_FORMAT != "csr":
        return edge_index, edge_weight
    # ATENCAO: com esparso nativo o PyG SOMA o self-loop (em vez de so'
    # completar os que faltam, como faz com edge_index); como i esta' na
    # propria lista (rks[i][0] = i), o no' ficaria com peso 2 em si mesmo.
    # Tirando os self-loops antes, a saida fica identica a do edge_index.
    # adj_t[j, i] = aresta i -> j: direcao da mensagem preservada.
    from torch_geometric.utils import to_torch_csr_tensor, remove_self_loops
    n = x.shape[0]
    ei, ew = remove_self_loops(edge_index, edge_weight)
    return to_torch_csr_tensor(ei.flip(0), ew, size=(n, n)), None


def fit(gcn_type, x, edge_index, labels, train_idx, seed, edge_weight=None, lr=LR,
        pseudo=None, w_p=1.0):
    """
    Treina e devolve (logp, hidden) de TODOS os nos, em numpy.

    pseudo: None ou (idx, classe, peso) dos nos pseudo-rotulados. Loss:
        nll(rotulados) + w_p * sum_i peso_i * nll_i(classe_i) / len(idx)
    Com peso = 1 e w_p = 1 em todos, e' o self-training classico com rotulo duro.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = x.device
    graph = _graph_arg(x, edge_index, edge_weight)
    n_classes = int(labels.max()) + 1
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    if pseudo is not None:
        p_idx = torch.as_tensor(pseudo[0], dtype=torch.long, device=device)
        p_y = torch.as_tensor(pseudo[1], dtype=torch.long, device=device)
        p_w = torch.as_tensor(pseudo[2], dtype=torch.float32, device=device)

    model = make_model(gcn_type, x.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        out = model(x, *graph)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        if pseudo is not None and len(p_idx):
            nll = F.nll_loss(out[p_idx], p_y, reduction="none")
            loss = loss + w_p * (p_w * nll).sum() / len(p_idx)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logp = model(x, *graph)
    return logp.cpu().numpy(), model.hidden.cpu().numpy()


def train_and_predict(gcn_type, x, edge_index, labels, train_idx, seed, edge_weight=None, lr=LR):
    """
    Treina com os rotulos de train_idx e devolve a predicao de TODOS os nos
    (numpy, shape n). Quem chama escolhe em quais nos avaliar.

    x, edge_index, edge_weight: tensores ja' no device (reusados entre runs).
    """
    logp, _ = fit(gcn_type, x, edge_index, labels, train_idx, seed, edge_weight, lr)
    return logp.argmax(axis=1)
