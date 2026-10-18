"""Load Cora, CiteSeer and PubMed from the original Planetoid files.

The files are the ones released with Yang, Cohen and Salakhutdinov (2016) at
github.com/kimiyoung/planetoid. They are Python 2 pickles; they are read with
an unpickler that only accepts the handful of types the files contain, and
each download is checked against a pinned SHA-256. The split is the public one
used by Kipf and Welling (2017): 20 labelled nodes per class for training, the
next 500 nodes for validation and 1,000 fixed test nodes.
"""

from __future__ import annotations

import collections
import hashlib
import os
import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:  # numpy < 2
    from numpy.core.multiarray import _reconstruct

URL = "https://github.com/kimiyoung/planetoid/raw/master/data/ind.{name}.{part}"
PARTS = ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index")
DATASETS = ("cora", "citeseer", "pubmed")
DEFAULT_ROOT = Path(os.environ.get("GCN_DATA", Path(__file__).resolve().parents[2] / "data"))

SHA256 = {
    "cora.x": "23cfa55d91c6f624233f5eb7b6e3f141f1bd2b2ae39608a63cf5d084bb27baab",
    "cora.y": "94465c14eb53e04ca262198dcbb2521ee8af60fb3f3d546cd6ca24a511b0e7d1",
    "cora.tx": "b9afbaa4a400df991f6f02ef677e1e44da55ffc04801fd02f9e673987829226a",
    "cora.ty": "41f5ac76596a1699cc33f53084a2419e92961c42bb75f36fc38e616d348532ad",
    "cora.allx": "9419ba2f26f5c35243db64aba110e0f35a04851609e0dc5433450676ca6b8543",
    "cora.ally": "2b998f5cc7fedc86e7f97ca2498f47a1ffc1462c29e2d578b23bf3f62b6e7d71",
    "cora.graph": "58f13302f39dde8852dad6fe6d15b89b077b6d7f837626bce671ef80344b383d",
    "cora.test.index": "297ce89af2b51a6a194181c7dbe1796c6a1cf9cd9349a88383b1b1e227867875",
    "citeseer.x": "d20de19150741e555de0ed037195630fc194114ebc7fd72ced62810148aa22f6",
    "citeseer.y": "8ba87e515d8e3ee3cf52f6d2c5b86aa73f11ecad825a808455209ec26cd54924",
    "citeseer.tx": "539a8906a2da97628d212f2fdd668c109a8b3f0d36574b59e5d3fd5b5303da21",
    "citeseer.ty": "55e5bd1ba1e733a04598753ad1a8d57957f4b8f89e74519f41f4f4938767e4ce",
    "citeseer.allx": "2ac30345d95c9ec933a817ee0bdd6a5f077f8c184a20e696910192d534414668",
    "citeseer.ally": "f704b2d986dde6c2669934de1f3ae5696a6cf9455c0f6b2646a2d135ea0a1c95",
    "citeseer.graph": "d79a4ef9d3e7169aee8946145f6b7306e35bc79bffad26346cfb94e10e9912a7",
    "citeseer.test.index": "2af990671580b6b2df5d158d1038f8292e30c9cd821b5453f399659b25b723d4",
    "pubmed.x": "fbe9cb5c47200d1b7769a26e579be3b7130b556831f9d850e77c4c74f2599b33",
    "pubmed.y": "e6c807633307a07ed659249006536a147c7999c797f11a2560752990181b41b3",
    "pubmed.tx": "4eb5f5d0f30f26497eb0ac6ac9719d09c67e284f1ed0385c93f3ef27a671da3d",
    "pubmed.ty": "ee2e0d4819d9fbcc403689c3bf2e49face401b605932f093bc58ab6461130fdd",
    "pubmed.allx": "508e538e2abdccdd54c4d3e9f49d638b4af7107d85845f66ddea5a5152658b7e",
    "pubmed.ally": "0c37bbe3b5014ec365e9d393cf4791925e5ed94fb194305d74f064a8d7b50f0e",
    "pubmed.graph": "5b89f0036ca22909471f1e6558eb42fa06e0c730a577bde5b058b40638f81dc9",
    "pubmed.test.index": "b101d421688bbaecbd82e8a18f1e282378d6830f3a37666f28ebc47f844341b5",
}

# Cora's label columns, matched to the class sizes of the original Cora release.
CORA_CLASSES = (
    "Theory",
    "Reinforcement learning",
    "Genetic algorithms",
    "Neural networks",
    "Probabilistic methods",
    "Case based",
    "Rule learning",
)


class _CSR:
    """Holds the state of a pickled scipy CSR matrix without importing scipy."""

    def __setstate__(self, state: dict) -> None:
        self.shape = tuple(state["_shape"])
        self.data = np.asarray(state["data"], dtype=np.float32)
        self.indices = np.asarray(state["indices"], dtype=np.int64)
        self.indptr = np.asarray(state["indptr"], dtype=np.int64)

    def dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=np.float32)
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        out[rows, self.indices] = self.data
        return out


_ALLOWED = {
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
    ("scipy.sparse.csr", "csr_matrix"): _CSR,
    ("__builtin__", "list"): list,
    ("collections", "defaultdict"): collections.defaultdict,
}


class _PlanetoidUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            msg = f"unexpected type in Planetoid file: {module}.{name}"
            raise pickle.UnpicklingError(msg) from None


@dataclass
class Graph:
    name: str
    features: torch.Tensor  # (n, f), rows normalised to sum to 1
    labels: torch.Tensor  # (n,), -1 where the files give no label
    edges: torch.Tensor  # (2, e) undirected edges stored in both directions, no self-loops
    train: torch.Tensor  # node indices
    val: torch.Tensor
    test: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.features.shape[0]

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1


def download(name: str, root: Path = DEFAULT_ROOT) -> Path:
    """Fetch the eight files for `name` into root/planetoid and verify them."""
    folder = Path(root) / "planetoid"
    folder.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        path = folder / f"ind.{name}.{part}"
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(URL.format(name=name, part=part), tmp)
            tmp.rename(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SHA256[f"{name}.{part}"]:
            path.unlink()
            raise OSError(f"checksum mismatch for {path.name}; deleted, run again")
    return folder


def _read(folder: Path, name: str, part: str):
    with open(folder / f"ind.{name}.{part}", "rb") as fh:
        obj = _PlanetoidUnpickler(fh, encoding="latin1").load()
    return obj.dense() if isinstance(obj, _CSR) else obj


def load(name: str, root: Path = DEFAULT_ROOT) -> Graph:
    """Load one dataset with the public split. Downloads it on first use."""
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    folder = download(name, root)
    x, y, tx, ty, allx, ally, graph = (
        _read(folder, name, part) for part in ("x", "y", "tx", "ty", "allx", "ally", "graph")
    )
    test_order = np.loadtxt(folder / f"ind.{name}.test.index", dtype=np.int64)
    test_sorted = np.sort(test_order)

    if name == "citeseer":
        # Some CiteSeer test nodes are isolated and missing from tx/ty:
        # give them zero rows so that node indices line up with the graph.
        full = np.arange(test_sorted.min(), test_sorted.max() + 1)
        tx_full = np.zeros((len(full), tx.shape[1]), dtype=tx.dtype)
        tx_full[test_sorted - full[0]] = tx
        ty_full = np.zeros((len(full), ty.shape[1]), dtype=ty.dtype)
        ty_full[test_sorted - full[0]] = ty
        tx, ty = tx_full, ty_full

    features = np.vstack([allx, tx]).astype(np.float32)
    onehot = np.vstack([ally, ty])
    # The test rows are stored in sorted order; put them at their graph indices.
    features[test_order] = features[test_sorted]
    onehot[test_order] = onehot[test_sorted]
    labels = np.where(onehot.sum(axis=1) > 0, onehot.argmax(axis=1), -1)

    rowsum = features.sum(axis=1, keepdims=True)
    features = np.divide(features, rowsum, out=np.zeros_like(features), where=rowsum > 0)

    n = features.shape[0]
    src, dst = [], []
    for node, neighbours in graph.items():
        for other in neighbours:
            if node != other and node < n and other < n:
                src.append(node)
                dst.append(other)
    pairs = np.unique(np.sort(np.array([src, dst]), axis=0), axis=1)
    edges = np.concatenate([pairs, pairs[::-1]], axis=1)

    n_train = len(y)
    return Graph(
        name=name,
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels.astype(np.int64)),
        edges=torch.from_numpy(edges.astype(np.int64)),
        train=torch.arange(n_train),
        val=torch.arange(n_train, n_train + 500),
        test=torch.from_numpy(test_sorted),
    )


def normalized_adjacency(edges: torch.Tensor, n: int) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 as a sparse tensor (the renormalisation trick, eq. 8)."""
    loops = torch.arange(n).repeat(2, 1)
    index = torch.cat([edges, loops], dim=1)
    degree = torch.bincount(index[0], minlength=n).float()
    scale = degree.pow(-0.5)
    values = scale[index[0]] * scale[index[1]]
    return torch.sparse_coo_tensor(index, values, (n, n), check_invariants=True).coalesce()
