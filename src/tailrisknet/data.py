"""Input contracts, validation, and deterministic data fingerprints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelData:
    returns: pd.DataFrame
    states: pd.DataFrame | None
    metadata: pd.DataFrame
    truth: pd.DataFrame | None = None


def load_panel(
    returns_path: Path,
    metadata_path: Path,
    states_path: Path | None = None,
    truth_path: Path | None = None,
) -> PanelData:
    """Load validated wide returns, optional state variables, and node metadata."""

    returns = _read_dated_csv(returns_path)
    states = _read_dated_csv(states_path) if states_path is not None else None
    metadata = pd.read_csv(metadata_path)
    truth = pd.read_csv(truth_path) if truth_path is not None and truth_path.exists() else None
    return validate_panel(returns, metadata, states, truth)


def validate_panel(
    returns: pd.DataFrame,
    metadata: pd.DataFrame,
    states: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> PanelData:
    """Validate and align research inputs without silently imputing observations."""

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("returns dates must be unique and sorted")
    if returns.columns.has_duplicates:
        raise ValueError("institution columns must be unique")
    if returns.shape[1] < 3:
        raise ValueError("at least three institutions are required")
    returns = returns.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if returns.isna().mean().max() > 0.10:
        raise ValueError("an institution has more than 10% missing returns")
    returns = returns.dropna(axis=0, how="any")
    if returns.empty or (returns.std(ddof=0) <= 0).any():
        raise ValueError("each institution must have non-constant observations")

    required = {"institution", "sector", "display_name", "market_cap"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if metadata["institution"].duplicated().any():
        raise ValueError("metadata institutions must be unique")
    metadata = metadata.set_index("institution", drop=False)
    absent = set(returns.columns) - set(metadata.index)
    if absent:
        raise ValueError(f"metadata missing for institutions: {sorted(absent)}")
    metadata = metadata.loc[list(returns.columns)].copy()
    metadata["market_cap"] = pd.to_numeric(metadata["market_cap"], errors="raise")
    if (metadata["market_cap"] <= 0).any():
        raise ValueError("market_cap must be positive")

    if states is not None:
        if not isinstance(states.index, pd.DatetimeIndex):
            raise ValueError("states index must be a DatetimeIndex")
        if states.index.has_duplicates or not states.index.is_monotonic_increasing:
            raise ValueError("state dates must be unique and sorted")
        states = states.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
        common = returns.index.intersection(states.index)
        returns = returns.loc[common]
        states = states.loc[common]
        valid = ~states.isna().any(axis=1)
        returns = returns.loc[valid]
        states = states.loc[valid]

    if truth is not None:
        expected = {"source", "target", "weight"}
        missing_truth = expected - set(truth.columns)
        if missing_truth:
            raise ValueError(f"truth data is missing columns: {sorted(missing_truth)}")
        unknown = (set(truth["source"]) | set(truth["target"])) - set(returns.columns)
        if unknown:
            raise ValueError(f"truth contains unknown institutions: {sorted(unknown)}")

    return PanelData(returns=returns, states=states, metadata=metadata, truth=truth)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_dated_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path.name} must contain a date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame.set_index("date").sort_index()
