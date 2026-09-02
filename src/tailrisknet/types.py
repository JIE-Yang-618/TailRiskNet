"""Small typed containers shared across estimation and reporting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NetworkEstimate:
    """One directed network estimate.

    Matrices use the convention ``rows = source`` and ``columns = target``.
    ``adjacency`` is non-negative adverse spillover magnitude. A negative entry
    in ``signed_effects`` means that moving the source from its median to its
    lower-tail VaR lowers the target's conditional quantile.
    """

    adjacency: pd.DataFrame
    signed_effects: pd.DataFrame
    coefficients: pd.DataFrame
    quantile: float
    alpha: float
    end_date: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        expected = (len(self.adjacency.index), len(self.adjacency.index))
        if self.adjacency.shape != expected:
            raise ValueError("adjacency must be square")
        if not self.adjacency.index.equals(self.adjacency.columns):
            raise ValueError("adjacency row and column labels must match")
        for matrix in (self.signed_effects, self.coefficients):
            if not matrix.index.equals(self.adjacency.index):
                raise ValueError("all network matrices must use the same labels")
            if not matrix.columns.equals(self.adjacency.columns):
                raise ValueError("all network matrices must use the same labels")
