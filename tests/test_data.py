import io
import pickle

import pytest
import torch

from gcn import load
from gcn.data import _PlanetoidUnpickler

# Nodes, features, classes, directed edge count, training nodes.
EXPECTED = {
    "cora": (2708, 1433, 7, 10556, 140),
    "citeseer": (3327, 3703, 6, 9104, 120),
    "pubmed": (19717, 500, 3, 88648, 60),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_shapes_and_split(name):
    nodes, features, classes, edges, n_train = EXPECTED[name]
    g = load(name)
    assert g.features.shape == (nodes, features)
    assert g.num_classes == classes
    assert g.edges.shape == (2, edges)
    assert len(g.train) == n_train and len(g.val) == 500 and len(g.test) == 1000
    split = torch.cat([g.train, g.val, g.test])
    assert len(torch.unique(split)) == len(split)
    # 20 labelled nodes per class for training
    assert torch.bincount(g.labels[g.train]).tolist() == [20] * classes
    assert (g.labels[g.test] >= 0).all()


@pytest.mark.parametrize("name", EXPECTED)
def test_graph_is_symmetric_without_self_loops(name):
    g = load(name)
    src, dst = g.edges
    assert (src != dst).all()
    forward = set(zip(src.tolist(), dst.tolist(), strict=True))
    assert all((b, a) in forward for a, b in forward)


def test_features_are_row_normalised():
    g = load("cora")
    sums = g.features.sum(dim=1)
    assert torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]), atol=1e-5)


def test_unpickler_rejects_other_types():
    payload = pickle.dumps(print, protocol=2)
    with pytest.raises(pickle.UnpicklingError):
        _PlanetoidUnpickler(io.BytesIO(payload), encoding="latin1").load()
