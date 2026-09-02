from __future__ import annotations

import numpy as np

from tailrisknet.estimators import (
    fit_pairwise_delta_covar,
    fit_penalized_quantile_network,
    fit_tail_uplift_network,
    rolling_quantile_network,
)
from tailrisknet.simulation import simulate_financial_panel


def test_penalized_network_has_explicit_orientation_and_zero_diagonal() -> None:
    panel = simulate_financial_panel(periods=180, institutions_per_sector=2, random_seed=11)
    estimate = fit_penalized_quantile_network(
        panel.returns,
        panel.states,
        quantile=0.05,
        alpha=0.01,
        factor_count=1,
    )

    assert list(estimate.adjacency.index) == list(panel.returns.columns)
    assert estimate.adjacency.index.equals(estimate.adjacency.columns)
    assert np.allclose(np.diag(estimate.adjacency), 0.0)
    assert (estimate.adjacency.to_numpy() >= 0).all()


def test_alternative_estimators_return_comparable_matrices() -> None:
    panel = simulate_financial_panel(periods=150, institutions_per_sector=2, random_seed=12)
    pairwise = fit_pairwise_delta_covar(panel.returns, panel.states, quantile=0.05, factor_count=1)
    uplift = fit_tail_uplift_network(panel.returns, quantile=0.05)

    assert pairwise.adjacency.shape == uplift.adjacency.shape == (6, 6)
    assert (uplift.adjacency.to_numpy() >= 0).all()


def test_rolling_windows_use_fixed_specification() -> None:
    panel = simulate_financial_panel(periods=180, institutions_per_sector=2, random_seed=13)
    estimates = rolling_quantile_network(
        panel.returns,
        panel.states,
        quantile=0.05,
        alpha=0.01,
        factor_count=1,
        window=120,
        step=30,
        min_observations=100,
    )

    assert len(estimates) == 3
    assert {estimate.alpha for estimate in estimates} == {0.01}
    assert estimates[-1].end_date == panel.returns.index[-1]
