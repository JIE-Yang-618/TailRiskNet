from __future__ import annotations

import numpy as np
import pandas as pd

from tailrisknet.metrics import (
    connectedness_summary,
    edge_list,
    node_metrics,
    recovery_metrics,
    sector_spillovers,
)


def _example() -> tuple[pd.DataFrame, pd.DataFrame]:
    adjacency = pd.DataFrame(
        [[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [3.0, 0.0, 0.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    metadata = pd.DataFrame(
        {
            "institution": ["A", "B", "C"],
            "sector": ["Bank", "Bank", "Insurance"],
            "display_name": ["A", "B", "C"],
            "market_cap": [2.0, 1.0, 1.0],
        }
    ).set_index("institution", drop=False)
    return adjacency, metadata


def test_source_row_metrics_are_not_reversed() -> None:
    adjacency, metadata = _example()
    metrics = node_metrics(adjacency, metadata).set_index("institution")

    assert metrics.loc["A", "out_strength"] == 2.0
    assert metrics.loc["A", "in_strength"] == 3.0
    assert metrics.loc["C", "net_transmitter"] == 2.0


def test_sector_means_normalize_possible_edge_count() -> None:
    adjacency, metadata = _example()
    sectors = sector_spillovers(adjacency, metadata)
    bank_to_bank = sectors.query("source_sector == 'Bank' and target_sector == 'Bank'").iloc[0]

    assert bank_to_bank["possible_edges"] == 2
    assert bank_to_bank["total_weight"] == 2.0
    assert bank_to_bank["mean_weight"] == 1.0


def test_connectedness_and_recovery() -> None:
    adjacency, _ = _example()
    summary = connectedness_summary(adjacency)
    truth = edge_list(adjacency)
    recovery = recovery_metrics(adjacency, truth)

    assert summary["total_connectedness"] == 6.0
    assert np.isclose(summary["density"], 0.5)
    assert recovery["precision"] == 1.0
    assert recovery["recall"] == 1.0
