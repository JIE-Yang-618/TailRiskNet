"""Dependence-aware uncertainty diagnostics for estimated network edges."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tailrisknet.estimators import fit_penalized_quantile_network


def moving_block_positions(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Draw circular moving blocks until ``length`` positions are obtained."""

    if length < 2:
        raise ValueError("length must be at least two")
    if not 1 <= block_length <= length:
        raise ValueError("block_length must lie between one and sample length")
    starts = rng.integers(0, length, size=int(np.ceil(length / block_length)))
    blocks = [(start + np.arange(block_length)) % length for start in starts]
    return np.concatenate(blocks)[:length]


def bootstrap_edge_stability(
    returns: pd.DataFrame,
    states: pd.DataFrame | None = None,
    *,
    quantile: float,
    alpha: float,
    factor_count: int,
    reps: int,
    block_length: int,
    lag: int = 1,
    edge_tolerance: float = 1e-6,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Estimate selection probabilities and bootstrap weight intervals.

    This is an edge-stability diagnostic, not a formal simultaneous confidence
    region. Moving blocks retain short-run serial dependence that an iid
    bootstrap would discard.
    """

    labels = list(returns.columns)
    if reps == 0:
        return _empty_stability(labels)
    rng = np.random.default_rng(random_seed)
    draws: list[np.ndarray] = []

    for _ in range(reps):
        positions = moving_block_positions(len(returns), block_length, rng)
        sampled_returns = returns.iloc[positions].copy()
        sampled_returns.index = pd.RangeIndex(len(sampled_returns))
        sampled_states = None
        if states is not None:
            sampled_states = states.iloc[positions].copy()
            sampled_states.index = sampled_returns.index
        estimate = fit_penalized_quantile_network(
            sampled_returns,
            sampled_states,
            quantile=quantile,
            alpha=alpha,
            factor_count=factor_count,
            lag=lag,
        )
        draws.append(estimate.adjacency.to_numpy())

    array = np.stack(draws, axis=0)
    rows: list[dict[str, float | str]] = []
    for source_index, source in enumerate(labels):
        for target_index, target in enumerate(labels):
            if source == target:
                continue
            values = array[:, source_index, target_index]
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "selection_probability": float(np.mean(values > edge_tolerance)),
                    "median_weight": float(np.median(values)),
                    "weight_p10": float(np.quantile(values, 0.10)),
                    "weight_p90": float(np.quantile(values, 0.90)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["selection_probability", "median_weight"], ascending=False, ignore_index=True
    )


def stability_filter(
    adjacency: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    minimum_probability: float,
) -> pd.DataFrame:
    """Keep only edges selected in enough moving-block bootstrap samples."""

    filtered = adjacency.copy()
    if stability.empty:
        return filtered
    probability = stability.set_index(["source", "target"])["selection_probability"]
    for source in adjacency.index:
        for target in adjacency.columns:
            if source == target:
                filtered.loc[source, target] = 0.0
                continue
            if float(probability.get((source, target), 0.0)) < minimum_probability:
                filtered.loc[source, target] = 0.0
    return filtered


def _empty_stability(labels: list[str]) -> pd.DataFrame:
    rows = [
        {
            "source": source,
            "target": target,
            "selection_probability": np.nan,
            "median_weight": np.nan,
            "weight_p10": np.nan,
            "weight_p90": np.nan,
        }
        for source in labels
        for target in labels
        if source != target
    ]
    return pd.DataFrame(rows)
