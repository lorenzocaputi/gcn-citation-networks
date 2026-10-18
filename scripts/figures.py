"""Make the README figures, in light and dark versions.

figures/tsne-cora.png    t-SNE of Cora's raw features next to the trained GCN's hidden layer
figures/accuracy.png     test accuracy per seed against the paper (needs results/accuracy.csv)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from gcn import load, normalized_adjacency, train
from gcn.data import CORA_CLASSES

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
STYLE = {
    "light": [FIG / "style.mplstyle"],
    "dark": [FIG / "style.mplstyle", FIG / "style-dark.mplstyle"],
}
INK = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
    },
}
# Kipf and Welling (2017), Table 2; their LP row quotes Yang et al. (2016).
PAPER = {
    "gcn": {"cora": 81.5, "citeseer": 70.3, "pubmed": 79.0},
    "label_propagation": {"cora": 68.0, "citeseer": 45.3, "pubmed": 63.0},
}
NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def save(fig, name: str, theme: str) -> None:
    suffix = "" if theme == "light" else "-dark"
    fig.savefig(FIG / f"{name}{suffix}.png")
    plt.close(fig)


def neighbour_agreement(points: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Share of each node's k nearest neighbours (cosine) that have its class."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(points)
    _, idx = nn.kneighbors(points)
    return float((labels[idx[:, 1:]] == labels[:, None]).mean())


def tsne_inputs():
    torch.manual_seed(0)
    graph = load("cora")
    adj = normalized_adjacency(graph.edges, graph.num_nodes)
    model, result = train(graph, "gcn", seed=0, adj=adj)
    model.eval()
    with torch.no_grad():
        hidden = model.hidden(graph.features.to_sparse(), adj).numpy()
    raw = graph.features.numpy()
    labels = graph.labels.numpy()
    raw_2d = TSNE(init="pca", perplexity=30, random_state=0).fit_transform(
        PCA(n_components=50, random_state=0).fit_transform(raw)
    )
    hidden_2d = TSNE(init="pca", perplexity=30, random_state=0).fit_transform(hidden)
    scores = (neighbour_agreement(raw, labels), neighbour_agreement(hidden, labels))
    summary = {
        "seed": 0,
        "test_acc": round(result.test_acc, 4),
        "knn10_same_class_raw": round(scores[0], 4),
        "knn10_same_class_hidden": round(scores[1], 4),
    }
    (ROOT / "results" / "tsne-cora.json").write_text(json.dumps(summary, indent=1) + "\n")
    return raw_2d, hidden_2d, labels, scores, result.test_acc


def tsne_figure(theme: str, data) -> None:
    raw_2d, hidden_2d, labels, (raw_score, hidden_score), test_acc = data
    plt.style.use(STYLE[theme])
    ink = INK[theme]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6), layout="constrained")
    panels = [
        (
            raw_2d,
            "Raw features",
            f"1,433-word bag of words · {raw_score:.0%} of 10 nearest neighbours share the class",
        ),
        (
            hidden_2d,
            "GCN hidden layer",
            f"16 units after training · {hidden_score:.0%} of 10"
            " nearest neighbours share the class",
        ),
    ]
    for ax, (points, title, subtitle) in zip(axes, panels, strict=True):
        for k, _name in enumerate(CORA_CLASSES):
            sel = labels == k
            ax.scatter(
                points[sel, 0],
                points[sel, 1],
                s=4,
                color=colors[k],
                linewidths=0,
                alpha=0.85,
                rasterized=True,
            )
        ax.set_title(title, fontsize=12)
        ax.text(
            0,
            1.0,
            subtitle,
            transform=ax.transAxes,
            fontsize=8,
            color=ink["secondary"],
            va="bottom",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # Direct labels at the class medians of the GCN map.
    halo = [pe.withStroke(linewidth=3, foreground=ink["surface"])]
    for k, name in enumerate(CORA_CLASSES):
        cx, cy = np.median(hidden_2d[labels == k], axis=0)
        axes[1].text(
            cx,
            cy,
            name,
            fontsize=8,
            ha="center",
            va="center",
            color=ink["primary"],
            path_effects=halo,
        )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=colors[k], label=name)
        for k, name in enumerate(CORA_CLASSES)
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=4, handletextpad=0.3, columnspacing=1.2
    )
    fig.suptitle(
        f"Cora, 2,708 papers coloured by topic. The GCN saw 140 labels "
        f"and scores {test_acc:.1%} on the test set.",
        x=0.01,
        ha="left",
        fontsize=11,
        color=ink["secondary"],
    )
    save(fig, "tsne-cora", theme)


def accuracy_figure(theme: str) -> None:
    runs = defaultdict(list)
    with open(ROOT / "results" / "accuracy.csv") as fh:
        for row in csv.DictReader(fh):
            runs[(row["dataset"], row["method"])].append(100 * float(row["test_acc"]))
    plt.style.use(STYLE[theme])
    ink = INK[theme]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    methods = [("gcn", "GCN"), ("mlp", "MLP"), ("label_propagation", "Label prop.")]
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9), sharey=True, layout="constrained")
    rng = np.random.default_rng(0)
    for ax, dataset in zip(axes, NAMES, strict=True):
        for i, (method, _label) in enumerate(methods):
            accs = np.array(runs[(dataset, method)])
            y = len(methods) - 1 - i
            jitter = rng.uniform(-0.12, 0.12, len(accs)) if len(accs) > 1 else np.zeros(1)
            ax.scatter(
                accs,
                y + jitter,
                s=22,
                color=colors[i],
                edgecolors=ink["surface"],
                linewidths=1,
                zorder=3,
            )
            ax.text(
                accs.max() + 1.2,
                y,
                f"{accs.mean():.1f}",
                ha="left",
                va="center",
                fontsize=9,
                color=ink["primary"],
            )
            if method in PAPER:
                paper = PAPER[method][dataset]
                ax.plot(
                    [paper, paper], [y - 0.32, y + 0.32], color=ink["primary"], lw=1.5, zorder=2
                )
        ax.set_title(NAMES[dataset], fontsize=12)
        ax.set_xlim(40, 90)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlabel("Test accuracy (%)")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(methods)), [m[1] for m in reversed(methods)])
    for ax in axes:
        ax.spines["left"].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=ink["muted"],
                   label="this repo, one dot per seed; number = mean"),
        plt.Line2D([], [], color=ink["primary"], linewidth=1.5,
                   label="Kipf & Welling (2017), Table 2"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    save(fig, "accuracy", theme)


def main() -> None:
    torch.set_num_threads(2)
    data = tsne_inputs()
    for theme in ("light", "dark"):
        tsne_figure(theme, data)
        if (ROOT / "results" / "accuracy.csv").exists():
            accuracy_figure(theme)


if __name__ == "__main__":
    main()
