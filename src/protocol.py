"""
Evaluation protocol of our group's previous rank-based GCN papers.

10 stratified folds WITHOUT shuffling; each fold (10% of n) is the labeled
set and the other 90% are the test set. Identical to the splits of the
original code, so the baseline reproduces the published tables.

The labeled set does NOT depend on the method: in each fold all methods
start from the same labels and are evaluated on the same test set (the
comparison is paired by fold).

Transductive: graph and ranked lists use all nodes, without labels; the
true labels of test nodes never enter training.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 10


def paper_folds(labels):
    """Lista de (train_idx, test_idx), um por fold. train = 10% de n."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)
    return [(np.sort(tr), np.sort(te)) for te, tr in skf.split(np.zeros(len(labels)), labels)]


# ---------------------------------------------------------------- active learning
# No active learning cada metodo escolhe NOS DIFERENTES pra rotular, entao o
# "resto" muda de metodo pra metodo. Por isso aqui o teste e' FIXO (20%,
# separado antes de tudo, nunca pode ser escolhido pra rotular) -- o mesmo
# protocolo fixed_test do codigo anterior do grupo. O rotulo so' e' revelado pros
# nos escolhidos (simula o anotador humano); o orcamento total e' 10% de n.

from sklearn.model_selection import StratifiedShuffleSplit

TEST_SIZE = 0.2


def fixed_test_split(labels, split_seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=split_seed)
    pool, test = next(sss.split(np.zeros(len(labels)), labels))
    return np.sort(pool), np.sort(test)
