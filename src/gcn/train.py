"""Training with the paper's settings, and the label-propagation baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import spsolve

from .data import Graph, normalized_adjacency
from .models import GCN, MLP


@dataclass
class Config:
    """Section 5.2 of the paper: Adam, lr 0.01, 200 epochs, dropout 0.5,
    L2 5e-4 on the first layer, 16 hidden units, early stopping window 10."""

    hidden: int = 16
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    window: int = 10


@dataclass
class Result:
    method: str
    dataset: str
    seed: int
    test_acc: float
    val_acc: float
    epochs: int


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def train(
    graph: Graph,
    method: str = "gcn",
    seed: int = 0,
    config: Config | None = None,
    adj: torch.Tensor | None = None,
) -> tuple[GCN, Result]:
    """Train a GCN or MLP once; returns the final model and its accuracies.

    Early stopping follows the reference implementation (github.com/tkipf/gcn):
    stop once the validation loss is above the mean of the previous `window`
    epochs. The test accuracy is that of the final weights.
    """
    config = config or Config()
    torch.manual_seed(seed)
    model_cls = {"gcn": GCN, "mlp": MLP}[method]
    model = model_cls(graph.features.shape[1], config.hidden, graph.num_classes, config.dropout)
    if method == "gcn" and adj is None:
        adj = normalized_adjacency(graph.edges, graph.num_nodes)
    if method == "mlp":
        adj = None
    regularized = {id(p) for p in model.regularized()}
    optimizer = torch.optim.Adam(
        [
            {"params": model.regularized(), "weight_decay": config.weight_decay},
            {"params": [p for p in model.parameters() if id(p) not in regularized]},
        ],
        lr=config.lr,
    )
    x, y = graph.features.to_sparse(), graph.labels
    # Evaluation has no dropout; a dense product is faster unless X is very sparse.
    x_eval = graph.features if x._nnz() > 0.05 * graph.features.numel() else x
    val_losses: list[float] = []
    epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, adj)[graph.train], y[graph.train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(x_eval, adj)[graph.val], y[graph.val]).item()
        val_losses.append(val_loss)
        recent = val_losses[-config.window - 1 : -1]
        if epoch > config.window and val_loss > sum(recent) / len(recent):
            break

    model.eval()
    with torch.no_grad():
        logits = model(x_eval, adj)
    result = Result(
        method=method,
        dataset=graph.name,
        seed=seed,
        test_acc=accuracy(logits[graph.test], y[graph.test]),
        val_acc=accuracy(logits[graph.val], y[graph.val]),
        epochs=epoch,
    )
    return model, result


def label_propagation(graph: Graph, class_mass: bool = True) -> Result:
    """Harmonic-function label propagation (Zhu, Ghahramani and Lafferty, 2003).

    Solves (D - A)_UU F_U = A_UL Y_L exactly for the unlabelled nodes U, using
    only the training labels as L. With `class_mass`, each class column is
    rescaled so that its total mass matches the class frequencies of the
    training labels (class mass normalisation, from the same paper). A node in
    a connected component without any training node gets no prediction and
    counts as an error.
    """
    n, k = graph.num_nodes, graph.num_classes
    src, dst = graph.edges.numpy()
    adj = sparse.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    train_nodes = graph.train.numpy()
    _, component = csgraph.connected_components(adj, directed=False)
    reachable = np.isin(component, component[train_nodes])
    labelled = np.zeros(n, dtype=bool)
    labelled[train_nodes] = True
    unknown = np.flatnonzero(reachable & ~labelled)

    seeds = np.zeros((len(train_nodes), k))
    seeds[np.arange(len(train_nodes)), graph.labels.numpy()[train_nodes]] = 1.0
    laplacian = sparse.diags(np.asarray(adj.sum(axis=1)).ravel()) - adj
    rhs = adj[unknown][:, train_nodes] @ seeds
    solution = spsolve(laplacian[unknown][:, unknown].tocsc(), rhs).reshape(len(unknown), k)
    if class_mass:
        solution = solution * (seeds.mean(axis=0) / solution.sum(axis=0))

    predicted = np.full(n, -1)
    predicted[train_nodes] = graph.labels.numpy()[train_nodes]
    predicted[unknown] = solution.argmax(axis=1)
    y = graph.labels.numpy()
    test, val = graph.test.numpy(), graph.val.numpy()
    return Result(
        method="label_propagation" if class_mass else "label_propagation_plain",
        dataset=graph.name,
        seed=0,
        test_acc=float((predicted[test] == y[test]).mean()),
        val_acc=float((predicted[val] == y[val]).mean()),
        epochs=0,
    )
