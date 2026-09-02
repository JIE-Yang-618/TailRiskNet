from __future__ import annotations

import pandas as pd
import pytest

from tailrisknet.data import validate_panel
from tailrisknet.simulation import simulate_financial_panel


def test_simulated_panel_satisfies_data_contract() -> None:
    panel = simulate_financial_panel(periods=140, institutions_per_sector=2, random_seed=7)
    validated = validate_panel(panel.returns, panel.metadata, panel.states, panel.truth)

    assert validated.returns.shape == (140, 6)
    assert validated.states is not None
    assert list(validated.metadata.index) == list(validated.returns.columns)


def test_metadata_must_cover_every_return_column() -> None:
    panel = simulate_financial_panel(periods=120, institutions_per_sector=2, random_seed=8)
    incomplete = panel.metadata.iloc[:-1]

    with pytest.raises(ValueError, match="metadata missing"):
        validate_panel(panel.returns, incomplete, panel.states)


def test_dates_must_be_unique_and_sorted() -> None:
    panel = simulate_financial_panel(periods=120, institutions_per_sector=2, random_seed=9)
    unsorted = panel.returns.sort_index(ascending=False)

    with pytest.raises(ValueError, match="unique and sorted"):
        validate_panel(unsorted, panel.metadata, panel.states)


def test_rows_with_small_amount_of_missing_data_are_dropped() -> None:
    panel = simulate_financial_panel(periods=120, institutions_per_sector=2, random_seed=10)
    returns = panel.returns.copy()
    returns.iloc[0, 0] = pd.NA
    validated = validate_panel(returns, panel.metadata, panel.states)

    assert len(validated.returns) == 119
