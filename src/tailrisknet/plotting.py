"""Publication-oriented, deterministic visual summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

SECTOR_COLORS = {
    "Bank": "#3B6FB6",
    "Securities": "#D9822B",
    "Insurance": "#4C9A6A",
}


def plot_connectedness(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(history["date"], history["mean_connectedness"], color="#274C77", linewidth=1.8)
    ax.set(
        title="Rolling lower-tail network connectedness",
        xlabel="Window end",
        ylabel="Mean edge weight",
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_network(
    adjacency: pd.DataFrame,
    metadata: pd.DataFrame,
    path: Path,
    *,
    top_edges: int = 30,
) -> None:
    graph = nx.DiGraph()
    for node in adjacency.index:
        graph.add_node(node, sector=str(metadata.loc[node, "sector"]))
    candidates = [
        (source, target, float(adjacency.loc[source, target]))
        for source in adjacency.index
        for target in adjacency.columns
        if source != target and adjacency.loc[source, target] > 0
    ]
    for source, target, weight in sorted(candidates, key=lambda item: item[2], reverse=True)[
        :top_edges
    ]:
        graph.add_edge(source, target, weight=weight)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    positions = nx.spring_layout(graph, seed=2026, weight="weight", k=0.9)
    colors = [
        SECTOR_COLORS.get(str(metadata.loc[node, "sector"]), "#777777") for node in graph.nodes
    ]
    strengths = adjacency.sum(axis=1) + adjacency.sum(axis=0)
    node_sizes = 650 + 2400 * strengths / max(float(strengths.max()), 1e-12)
    edge_weights = [float(graph[u][v]["weight"]) for u, v in graph.edges]
    max_weight = max(edge_weights, default=1.0)
    widths = [0.7 + 3.3 * value / max_weight for value in edge_weights]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=colors,
        node_size=[float(node_sizes.loc[node]) for node in graph.nodes],
        edgecolors="white",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        graph,
        positions,
        width=widths,
        alpha=0.68,
        edge_color="#4E5968",
        arrows=True,
        arrowsize=18,
        min_source_margin=18,
        min_target_margin=22,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    ax.set_title("Stability-filtered lower-tail spillover network", pad=14)
    present_sectors = list(dict.fromkeys(metadata.loc[list(graph.nodes), "sector"].astype(str)))
    ax.legend(
        handles=[
            Patch(facecolor=SECTOR_COLORS.get(sector, "#777777"), label=sector)
            for sector in present_sectors
        ],
        loc="upper left",
        frameon=False,
        title="Sector",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sector_heatmap(sector_table: pd.DataFrame, path: Path) -> None:
    matrix = sector_table.pivot(
        index="source_sector", columns="target_sector", values="mean_weight"
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        matrix,
        cmap="mako_r",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Mean adverse edge weight"},
        ax=ax,
    )
    ax.set(
        xlabel="Receiving sector", ylabel="Emitting sector", title="Sector-normalized spillovers"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stability(stability: pd.DataFrame, labels: list[str], path: Path) -> None:
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for row in stability.itertuples():
        matrix.loc[row.source, row.target] = row.selection_probability
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    sns.heatmap(
        matrix,
        cmap="crest",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Bootstrap selection probability"},
        ax=ax,
    )
    ax.set(xlabel="Receiver", ylabel="Emitter", title="Moving-block bootstrap edge stability")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
