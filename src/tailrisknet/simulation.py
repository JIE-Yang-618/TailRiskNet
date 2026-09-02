"""A reproducible heavy-tailed panel with a known directed contagion graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulatedPanel:
    returns: pd.DataFrame
    states: pd.DataFrame
    metadata: pd.DataFrame
    truth: pd.DataFrame


def simulate_financial_panel(
    *,
    periods: int = 520,
    institutions_per_sector: int = 4,
    random_seed: int = 2026,
) -> SimulatedPanel:
    """Generate weekly returns with factors, fat tails, and asymmetric contagion.

    The DGP is intentionally stylized. It exists to test whether a method can
    recover known directional lower-tail links before it is applied to
    proprietary data where the true network is unobserved.
    """

    if periods < 100:
        raise ValueError("periods must be at least 100")
    if institutions_per_sector < 2:
        raise ValueError("institutions_per_sector must be at least two")

    rng = np.random.default_rng(random_seed)
    sectors = ("Bank", "Securities", "Insurance")
    prefixes = ("B", "S", "I")
    labels = [
        f"{prefix}{position + 1}"
        for prefix in prefixes
        for position in range(institutions_per_sector)
    ]
    node_count = len(labels)
    dates = pd.date_range("2014-01-03", periods=periods, freq="W-FRI")

    market = rng.standard_t(df=5, size=periods) * 0.009
    sector_factors = rng.standard_t(df=6, size=(periods, len(sectors))) * 0.0045
    liquidity = np.zeros(periods)
    for time in range(1, periods):
        liquidity[time] = 0.85 * liquidity[time - 1] + rng.normal(scale=0.15)

    betas = rng.uniform(0.65, 1.20, size=node_count)
    returns = np.zeros((periods, node_count))
    edge_specs = _edge_specification(institutions_per_sector)

    for time in range(periods):
        innovations = rng.standard_t(df=4, size=node_count) * 0.007
        base = np.empty(node_count)
        for node in range(node_count):
            sector_index = node // institutions_per_sector
            base[node] = (
                betas[node] * market[time] + sector_factors[time, sector_index] + innovations[node]
            )
            if time > 0:
                base[node] += 0.08 * returns[time - 1, node]

        # A source tail event at t-1 affects the target at t, creating an
        # observable temporal direction rather than symmetric contemporaneous
        # comovement.
        if time > 0:
            for source, target, weight in edge_specs:
                tail_excess = min(returns[time - 1, source] + 0.008, 0.0)
                base[target] += weight * tail_excess
        returns[time] = base

    return_frame = pd.DataFrame(returns, index=dates, columns=labels)
    state_frame = pd.DataFrame(
        {
            "market_return_lag1": pd.Series(market, index=dates).shift(1).fillna(0.0),
            "market_abs_4w": pd.Series(np.abs(market), index=dates)
            .rolling(4, min_periods=1)
            .mean(),
            "liquidity_state_lag1": pd.Series(liquidity, index=dates).shift(1).fillna(0.0),
        },
        index=dates,
    )

    metadata_rows: list[dict[str, float | str]] = []
    for sector_index, (sector, prefix) in enumerate(zip(sectors, prefixes, strict=True)):
        for position in range(institutions_per_sector):
            label = f"{prefix}{position + 1}"
            metadata_rows.append(
                {
                    "institution": label,
                    "sector": sector,
                    "display_name": f"Synthetic {sector} {position + 1}",
                    "market_cap": float(np.exp(rng.normal(4.5 + 0.25 * (sector_index == 0), 0.35))),
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    truth = pd.DataFrame(
        [
            {"source": labels[source], "target": labels[target], "weight": weight}
            for source, target, weight in edge_specs
        ]
    )
    return SimulatedPanel(return_frame, state_frame, metadata, truth)


def write_simulated_panel(panel: SimulatedPanel, directory: str | Path) -> None:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    panel.returns.rename_axis("date").to_csv(output / "returns.csv")
    panel.states.rename_axis("date").to_csv(output / "states.csv")
    panel.metadata.to_csv(output / "institutions.csv", index=False)
    panel.truth.to_csv(output / "true_edges.csv", index=False)


def _edge_specification(institutions_per_sector: int) -> list[tuple[int, int, float]]:
    second = institutions_per_sector
    third = 2 * institutions_per_sector
    edges = [
        (0, 1, 1.30),
        (second, second + 1, 1.20),
        (third, third + 1, 1.10),
        (1, second + 1, 0.90),
        (second + 1, third + 1, 0.85),
    ]
    if institutions_per_sector >= 3:
        edges.extend(
            [
                (1, 2, 1.00),
                (second + 1, second + 2, 0.95),
                (2, second + 2, 0.80),
                (second + 2, third + 2, 0.75),
            ]
        )
    return edges
