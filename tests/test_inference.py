from __future__ import annotations

import numpy as np

from tailrisknet.estimators import fit_penalized_quantile_network
from tailrisknet.inference import (
    bootstrap_edge_stability,
    moving_block_positions,
    stability_filter,
)
from tailrisknet.simulation import simulate_financial_panel


def test_moving_block_positions_are_valid_and_reproducible() -> None:
    first = moving_block_positions(25, 4, np.random.default_rng(42))
    second = moving_block_positions(25, 4, np.random.default_rng(42))

    assert np.array_equal(first, second)
    assert len(first) == 25
    assert first.min() >= 0 and first.max() < 25


def test_stability_probabilities_and_filter() -> None:
    panel = simulate_financial_panel(periods=130, institutions_per_sector=2, random_seed=14)
    estimate = fit_penalized_quantile_network(
        panel.returns, panel.states, quantile=0.05, alpha=0.01, factor_count=1
    )
    stability = bootstrap_edge_stability(
        panel.returns,
        panel.states,
        quantile=0.05,
        alpha=0.01,
        factor_count=1,
        reps=4,
        block_length=6,
        random_seed=14,
    )
    filtered = stability_filter(estimate.adjacency, stability, minimum_probability=0.75)

    assert len(stability) == 6 * 5
    assert stability["selection_probability"].between(0, 1).all()
    assert (filtered.to_numpy() <= estimate.adjacency.to_numpy()).all()
