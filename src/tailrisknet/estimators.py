"""Econometric estimators for directed lower-tail spillover networks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit

from tailrisknet.types import NetworkEstimate


def fit_penalized_quantile_network(
    returns: pd.DataFrame,
    states: pd.DataFrame | None = None,
    *,
    quantile: float = 0.05,
    alpha: float = 0.01,
    factor_count: int = 1,
    lag: int = 1,
) -> NetworkEstimate:
    """Estimate a sparse, directed, adverse lower-tail network.

    Each target institution is regressed at ``quantile`` on lagged returns of
    all other institutions and optional predetermined state variables.
    L1-regularized quantile regression performs edge selection.

    The edge ``source -> target`` is the adverse change in the target's fitted
    conditional quantile when the standardized source moves from its median to
    its lower-tail VaR. The model identifies conditional tail associations, not
    structural or causal transmission.
    """

    _validate_estimation_inputs(returns, states, quantile, factor_count, lag)
    adjusted = residualize_common_factors(returns, factor_count)
    standardized, _, _ = _standardize(adjusted)
    state_standardized = _standardize(states)[0] if states is not None else None
    sources, targets, aligned_states = _lagged_frames(standardized, state_standardized, lag)
    labels = list(returns.columns)
    adjacency = pd.DataFrame(0.0, index=labels, columns=labels)
    signed = adjacency.copy()
    coefficients = adjacency.copy()

    stress_moves = sources.quantile(quantile) - sources.median()
    for target in labels:
        peers = [name for name in labels if name != target]
        features = sources[peers].copy()
        features["__own_lag__"] = sources[target]
        if aligned_states is not None:
            features = pd.concat([features, aligned_states], axis=1)
        model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
        model.fit(features.to_numpy(), targets[target].to_numpy())
        peer_coefficients = pd.Series(model.coef_[: len(peers)], index=peers)
        effects = peer_coefficients * stress_moves.loc[peers]
        coefficients.loc[peers, target] = peer_coefficients
        signed.loc[peers, target] = effects
        adjacency.loc[peers, target] = (-effects).clip(lower=0.0)

    return NetworkEstimate(
        adjacency=adjacency,
        signed_effects=signed,
        coefficients=coefficients,
        quantile=quantile,
        alpha=alpha,
        end_date=pd.Timestamp(returns.index[-1]),
    )


def fit_pairwise_delta_covar(
    returns: pd.DataFrame,
    states: pd.DataFrame | None = None,
    *,
    quantile: float = 0.05,
    factor_count: int = 0,
    lag: int = 0,
) -> NetworkEstimate:
    """Estimate a transparent pairwise Delta-CoVaR benchmark network."""

    _validate_estimation_inputs(returns, states, quantile, factor_count, lag)
    adjusted = residualize_common_factors(returns, factor_count)
    standardized, _, _ = _standardize(adjusted)
    state_standardized = _standardize(states)[0] if states is not None else None
    sources, targets, aligned_states = _lagged_frames(standardized, state_standardized, lag)
    labels = list(returns.columns)
    adjacency = pd.DataFrame(0.0, index=labels, columns=labels)
    signed = adjacency.copy()
    coefficients = adjacency.copy()
    stress_moves = sources.quantile(quantile) - sources.median()

    for source in labels:
        for target in labels:
            if source == target:
                continue
            features = sources[[source]].copy()
            if lag > 0:
                features["__own_lag__"] = sources[target]
            if aligned_states is not None:
                features = pd.concat([features, aligned_states], axis=1)
            model = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
            model.fit(features.to_numpy(), targets[target].to_numpy())
            beta = float(model.coef_[0])
            effect = beta * float(stress_moves[source])
            coefficients.loc[source, target] = beta
            signed.loc[source, target] = effect
            adjacency.loc[source, target] = max(0.0, -effect)

    return NetworkEstimate(
        adjacency=adjacency,
        signed_effects=signed,
        coefficients=coefficients,
        quantile=quantile,
        alpha=0.0,
        end_date=pd.Timestamp(returns.index[-1]),
    )


def fit_tail_uplift_network(
    returns: pd.DataFrame,
    *,
    quantile: float = 0.05,
    prior: float = 0.5,
    lag: int = 0,
) -> NetworkEstimate:
    """Estimate a distribution-free lower-tail co-exceedance diagnostic.

    The weight is the positive uplift in the probability that the target is in
    its lower tail conditional on the source being in its lower tail. Jeffreys
    smoothing (``prior=0.5``) prevents zero-probability artifacts.
    """

    _validate_estimation_inputs(returns, None, quantile, 0, lag)
    labels = list(returns.columns)
    events = returns.le(returns.quantile(quantile))
    adjacency = pd.DataFrame(0.0, index=labels, columns=labels)
    signed = adjacency.copy()

    for source in labels:
        source_events = events[source].shift(lag).dropna().astype(bool)
        target_events = events.loc[source_events.index]
        denominator = float(source_events.sum()) + 2 * prior
        for target in labels:
            if source == target:
                continue
            joint = float((source_events & target_events[target]).sum())
            conditional = (joint + prior) / denominator
            marginal = (float(target_events[target].sum()) + prior) / (
                len(target_events) + 2 * prior
            )
            uplift = conditional - marginal
            signed.loc[source, target] = uplift
            adjacency.loc[source, target] = max(0.0, uplift)

    return NetworkEstimate(
        adjacency=adjacency,
        signed_effects=signed,
        coefficients=signed.copy(),
        quantile=quantile,
        alpha=0.0,
        end_date=pd.Timestamp(returns.index[-1]),
    )


def rolling_quantile_network(
    returns: pd.DataFrame,
    states: pd.DataFrame | None = None,
    *,
    quantile: float,
    alpha: float,
    factor_count: int,
    window: int,
    step: int,
    min_observations: int,
    lag: int = 1,
) -> list[NetworkEstimate]:
    """Estimate fixed-specification networks over overlapping time windows."""

    if len(returns) < window:
        raise ValueError("returns contain fewer rows than the rolling window")
    estimates: list[NetworkEstimate] = []
    for end in range(window, len(returns) + 1, step):
        start = end - window
        return_window = returns.iloc[start:end]
        state_window = states.iloc[start:end] if states is not None else None
        valid = ~return_window.isna().any(axis=1)
        if state_window is not None:
            valid &= ~state_window.isna().any(axis=1)
        if int(valid.sum()) < min_observations:
            continue
        estimates.append(
            fit_penalized_quantile_network(
                return_window.loc[valid],
                None if state_window is None else state_window.loc[valid],
                quantile=quantile,
                alpha=alpha,
                factor_count=factor_count,
                lag=lag,
            )
        )
    if not estimates:
        raise ValueError("no rolling window met min_observations")
    return estimates


def select_alpha_time_series_cv(
    returns: pd.DataFrame,
    states: pd.DataFrame | None,
    candidates: Iterable[float],
    *,
    quantile: float,
    factor_count: int,
    splits: int = 3,
    lag: int = 1,
) -> tuple[float, pd.DataFrame]:
    """Select one penalty on a calibration window using expanding time folds.

    A single pre-calibrated penalty is then held fixed across rolling windows,
    so changes in connectedness are not mechanically induced by a changing
    regularization parameter.
    """

    candidate_values = tuple(float(value) for value in candidates)
    if not candidate_values:
        raise ValueError("at least one alpha candidate is required")
    splitter = TimeSeriesSplit(n_splits=splits)
    rows: list[dict[str, float | int]] = []
    labels = list(returns.columns)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(returns), start=1):
        train_returns = returns.iloc[train_idx]
        test_returns = returns.iloc[test_idx]
        train_adjusted, test_adjusted = residualize_train_test(
            train_returns, test_returns, factor_count
        )
        train_z, means, scales = _standardize(train_adjusted)
        test_z = (test_adjusted - means) / scales

        train_states: pd.DataFrame | None = None
        test_states: pd.DataFrame | None = None
        if states is not None:
            train_states, state_means, state_scales = _standardize(states.iloc[train_idx])
            test_states = (states.iloc[test_idx] - state_means) / state_scales

        train_sources, train_targets, train_states_aligned = _lagged_frames(
            train_z, train_states, lag
        )
        test_sources = _test_lagged_sources(train_z, test_z, lag)

        for alpha in candidate_values:
            losses: list[float] = []
            for target in labels:
                peers = [name for name in labels if name != target]
                x_train = train_sources[peers].copy()
                x_test = test_sources[peers].copy()
                x_train["__own_lag__"] = train_sources[target]
                x_test["__own_lag__"] = test_sources[target]
                if train_states_aligned is not None and test_states is not None:
                    x_train = pd.concat([x_train, train_states_aligned], axis=1)
                    x_test = pd.concat([x_test, test_states], axis=1)
                model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
                model.fit(x_train.to_numpy(), train_targets[target].to_numpy())
                prediction = model.predict(x_test.to_numpy())
                losses.append(mean_pinball_loss(test_z[target], prediction, alpha=quantile))
            rows.append({"fold": fold, "alpha": alpha, "pinball_loss": float(np.mean(losses))})

    scores = pd.DataFrame(rows)
    mean_scores = pd.DataFrame(scores.groupby("alpha")["pinball_loss"].mean().reset_index())
    best = float(mean_scores.sort_values(["pinball_loss", "alpha"]).iloc[0]["alpha"])
    return best, scores


def residualize_common_factors(returns: pd.DataFrame, factor_count: int) -> pd.DataFrame:
    """Remove contemporaneous PCA factors while preserving labels and dates."""

    if factor_count == 0:
        return returns.copy()
    if factor_count >= min(returns.shape):
        raise ValueError("factor_count must be smaller than both panel dimensions")
    model = PCA(n_components=factor_count, svd_solver="full")
    values = returns.to_numpy()
    fitted = model.inverse_transform(model.fit_transform(values))
    residuals = values - fitted + values.mean(axis=0)
    return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)


def residualize_train_test(
    train: pd.DataFrame, test: pd.DataFrame, factor_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit factor adjustment on training data and apply it to a later test fold."""

    if factor_count == 0:
        return train.copy(), test.copy()
    model = PCA(n_components=factor_count, svd_solver="full")
    train_values = train.to_numpy()
    test_values = test.to_numpy()
    model.fit(train_values)
    train_fitted = model.inverse_transform(model.transform(train_values))
    test_fitted = model.inverse_transform(model.transform(test_values))
    train_residuals = train_values - train_fitted + train_values.mean(axis=0)
    test_residuals = test_values - test_fitted + train_values.mean(axis=0)
    return (
        pd.DataFrame(train_residuals, index=train.index, columns=train.columns),
        pd.DataFrame(test_residuals, index=test.index, columns=test.columns),
    )


def _standardize(
    frame: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if frame is None:
        raise ValueError("cannot standardize None")
    means = frame.mean()
    scales = frame.std(ddof=0).replace(0.0, 1.0)
    return (frame - means) / scales, means, scales


def _lagged_frames(
    standardized: pd.DataFrame,
    states: pd.DataFrame | None,
    lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    sources = standardized.shift(lag) if lag else standardized.copy()
    valid_index = sources.dropna(axis=0, how="any").index
    aligned_states = states.loc[valid_index] if states is not None else None
    return sources.loc[valid_index], standardized.loc[valid_index], aligned_states


def _test_lagged_sources(train: pd.DataFrame, test: pd.DataFrame, lag: int) -> pd.DataFrame:
    if lag == 0:
        return test.copy()
    history = pd.concat([train.tail(lag), test])
    return history.shift(lag).loc[test.index]


def _validate_estimation_inputs(
    returns: pd.DataFrame,
    states: pd.DataFrame | None,
    quantile: float,
    factor_count: int,
    lag: int,
) -> None:
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must lie strictly between 0 and 0.5")
    if returns.isna().any().any():
        raise ValueError("estimation inputs cannot contain missing returns")
    if states is not None:
        if not states.index.equals(returns.index):
            raise ValueError("states and returns must have identical indices")
        if states.isna().any().any():
            raise ValueError("estimation inputs cannot contain missing states")
    if factor_count < 0:
        raise ValueError("factor_count cannot be negative")
    if not 0 <= lag < len(returns):
        raise ValueError("lag must be non-negative and smaller than the sample")
