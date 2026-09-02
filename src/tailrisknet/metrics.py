"""Transparent network summaries with an explicit edge orientation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def edge_list(adjacency: pd.DataFrame, *, tolerance: float = 1e-12) -> pd.DataFrame:
    rows = [
        {"source": source, "target": target, "weight": float(adjacency.loc[source, target])}
        for source in adjacency.index
        for target in adjacency.columns
        if source != target and adjacency.loc[source, target] > tolerance
    ]
    if not rows:
        return pd.DataFrame(columns=["source", "target", "weight"])
    return pd.DataFrame(rows).sort_values("weight", ascending=False, ignore_index=True)


def node_metrics(adjacency: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute emitter, receiver, and size-adjusted importance measures."""

    labels = list(adjacency.index)
    out_strength = adjacency.sum(axis=1)
    in_strength = adjacency.sum(axis=0)
    out_degree = (adjacency > 0).sum(axis=1)
    in_degree = (adjacency > 0).sum(axis=0)
    size_share = metadata.loc[labels, "market_cap"] / metadata.loc[labels, "market_cap"].sum()
    counterparty_size = size_share.to_numpy()
    size_weighted_emitter = size_share * adjacency.mul(counterparty_size, axis=1).sum(axis=1)
    size_weighted_receiver = size_share * adjacency.mul(counterparty_size, axis=0).sum(axis=0)

    result = pd.DataFrame(
        {
            "institution": labels,
            "sector": metadata.loc[labels, "sector"].to_numpy(),
            "display_name": metadata.loc[labels, "display_name"].to_numpy(),
            "out_strength": out_strength.to_numpy(),
            "in_strength": in_strength.to_numpy(),
            "net_transmitter": (out_strength - in_strength).to_numpy(),
            "out_degree": out_degree.to_numpy(),
            "in_degree": in_degree.to_numpy(),
            "market_cap_share": size_share.to_numpy(),
            "size_weighted_emitter": size_weighted_emitter.to_numpy(),
            "size_weighted_receiver": size_weighted_receiver.to_numpy(),
        }
    )
    return result.sort_values("out_strength", ascending=False, ignore_index=True)


def sector_spillovers(adjacency: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sector blocks by both total and exposure-count-normalized mean.

    Reporting the mean alongside the total prevents a sector with more listed
    institutions from appearing mechanically more systemic merely because it
    contains more possible directed edges.
    """

    sectors = list(dict.fromkeys(metadata.loc[adjacency.index, "sector"].astype(str)))
    rows: list[dict[str, float | int | str]] = []
    for source_sector in sectors:
        sources = metadata.index[metadata["sector"] == source_sector].intersection(adjacency.index)
        for target_sector in sectors:
            targets = metadata.index[metadata["sector"] == target_sector].intersection(
                adjacency.columns
            )
            values = [
                float(adjacency.loc[source, target])
                for source in sources
                for target in targets
                if source != target
            ]
            rows.append(
                {
                    "source_sector": source_sector,
                    "target_sector": target_sector,
                    "possible_edges": len(values),
                    "active_edges": int(np.sum(np.asarray(values) > 0)),
                    "total_weight": float(np.sum(values)),
                    "mean_weight": float(np.mean(values)) if values else 0.0,
                    "active_share": float(np.mean(np.asarray(values) > 0)) if values else 0.0,
                }
            )
    return pd.DataFrame(rows)


def connectedness_summary(adjacency: pd.DataFrame) -> dict[str, float | int]:
    off_diagonal = adjacency.to_numpy()[~np.eye(len(adjacency), dtype=bool)]
    return {
        "total_connectedness": float(off_diagonal.sum()),
        "mean_connectedness": float(off_diagonal.mean()),
        "active_edges": int(np.sum(off_diagonal > 0)),
        "density": float(np.mean(off_diagonal > 0)),
    }


def compare_networks(networks: dict[str, pd.DataFrame], *, top_k: int = 20) -> pd.DataFrame:
    """Pairwise rank correlations and top-edge overlap across estimators."""

    names = list(networks)
    rows: list[dict[str, float | int | str]] = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = networks[left_name]
            right = networks[right_name]
            mask = ~np.eye(len(left), dtype=bool)
            left_values = left.to_numpy()[mask]
            right_values = right.to_numpy()[mask]
            if np.allclose(left_values, left_values[0]) or np.allclose(
                right_values, right_values[0]
            ):
                correlation = np.nan
            else:
                correlation = spearmanr(left_values, right_values).statistic
            left_edges = {
                (row.source, row.target) for row in edge_list(left).head(top_k).itertuples()
            }
            right_edges = {
                (row.source, row.target) for row in edge_list(right).head(top_k).itertuples()
            }
            union = left_edges | right_edges
            rows.append(
                {
                    "model_a": left_name,
                    "model_b": right_name,
                    "spearman_weight_correlation": float(correlation),
                    "top_k": top_k,
                    "top_edge_jaccard": len(left_edges & right_edges) / len(union)
                    if union
                    else 1.0,
                }
            )
    return pd.DataFrame(rows)


def recovery_metrics(
    adjacency: pd.DataFrame, truth: pd.DataFrame, *, tolerance: float = 1e-12
) -> dict[str, float | int]:
    """Evaluate simulated edge recovery when a known data-generating graph exists."""

    truth_weights = pd.to_numeric(truth["weight"], errors="raise").to_numpy(dtype=float)
    truth_pairs = {
        (str(source), str(target))
        for source, target, weight in zip(
            truth["source"].astype(str),
            truth["target"].astype(str),
            truth_weights,
            strict=True,
        )
        if weight > 0
    }
    labels = list(adjacency.index)
    pairs = [(source, target) for source in labels for target in labels if source != target]
    y_true = np.array([int(pair in truth_pairs) for pair in pairs])
    scores = np.array([adjacency.loc[source, target] for source, target in pairs])
    selected = scores > tolerance
    true_positive = int(np.sum(selected & (y_true == 1)))
    false_positive = int(np.sum(selected & (y_true == 0)))
    false_negative = int(np.sum(~selected & (y_true == 1)))
    precision = true_positive / (true_positive + false_positive) if selected.any() else 0.0
    recall = true_positive / (true_positive + false_negative) if y_true.any() else 0.0
    return {
        "true_edges": int(y_true.sum()),
        "selected_edges": int(selected.sum()),
        "precision": precision,
        "recall": recall,
        "average_precision": float(average_precision_score(y_true, scores)),
    }
