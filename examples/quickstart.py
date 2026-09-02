"""Minimal API example using the known-graph simulation."""

from tailrisknet.estimators import fit_penalized_quantile_network
from tailrisknet.metrics import edge_list
from tailrisknet.simulation import simulate_financial_panel

panel = simulate_financial_panel(periods=260, institutions_per_sector=3, random_seed=2026)
estimate = fit_penalized_quantile_network(
    panel.returns,
    panel.states,
    quantile=0.05,
    alpha=0.015,
    factor_count=1,
)

print(edge_list(estimate.adjacency).head(10).to_string(index=False))
