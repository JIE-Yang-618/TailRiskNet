"""End-to-end research pipeline and auditable output manifest."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pandas as pd

from tailrisknet import __version__
from tailrisknet.config import ProjectConfig
from tailrisknet.data import file_sha256, load_panel
from tailrisknet.estimators import (
    fit_pairwise_delta_covar,
    fit_tail_uplift_network,
    rolling_quantile_network,
    select_alpha_time_series_cv,
)
from tailrisknet.inference import bootstrap_edge_stability, stability_filter
from tailrisknet.metrics import (
    compare_networks,
    connectedness_summary,
    edge_list,
    node_metrics,
    recovery_metrics,
    sector_spillovers,
)
from tailrisknet.plotting import (
    plot_connectedness,
    plot_network,
    plot_sector_heatmap,
    plot_stability,
)


def run_pipeline(config: ProjectConfig) -> dict[str, object]:
    panel = load_panel(
        config.data.returns,
        config.data.metadata,
        config.data.states,
        config.data.truth,
    )
    if len(panel.returns) < config.model.window:
        raise ValueError("validated panel is shorter than the configured rolling window")

    output = config.output.directory
    figure_directory = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    calibration = panel.returns.iloc[: config.model.window]
    calibration_states = (
        panel.states.iloc[: config.model.window] if panel.states is not None else None
    )
    alpha_cv: pd.DataFrame | None = None
    if config.model.alpha is None:
        alpha, alpha_cv = select_alpha_time_series_cv(
            calibration,
            calibration_states,
            config.model.alpha_grid,
            quantile=config.model.quantile,
            factor_count=config.model.factor_count,
            lag=config.model.lag,
        )
        alpha_cv.to_csv(output / "alpha_cv_scores.csv", index=False)
    else:
        alpha = config.model.alpha

    estimates = rolling_quantile_network(
        panel.returns,
        panel.states,
        quantile=config.model.quantile,
        alpha=alpha,
        factor_count=config.model.factor_count,
        window=config.model.window,
        step=config.model.step,
        min_observations=config.model.min_observations,
        lag=config.model.lag,
    )
    history_rows = [
        {"date": estimate.end_date, **connectedness_summary(estimate.adjacency)}
        for estimate in estimates
    ]
    history = pd.DataFrame(history_rows)
    history.to_csv(output / "connectedness.csv", index=False)

    latest = estimates[-1]
    latest.adjacency.to_csv(output / "latest_adjacency.csv", index_label="source")
    latest.signed_effects.to_csv(output / "latest_signed_effects.csv", index_label="source")

    latest_returns = panel.returns.iloc[-config.model.window :]
    latest_states = panel.states.iloc[-config.model.window :] if panel.states is not None else None
    stability = bootstrap_edge_stability(
        latest_returns,
        latest_states,
        quantile=config.model.quantile,
        alpha=alpha,
        factor_count=config.model.factor_count,
        lag=config.model.lag,
        reps=config.inference.bootstrap_reps,
        block_length=config.inference.block_length,
        edge_tolerance=config.model.edge_tolerance,
        random_seed=config.inference.random_seed,
    )
    stability.to_csv(output / "edge_stability.csv", index=False)
    filtered = stability_filter(
        latest.adjacency,
        stability,
        minimum_probability=config.inference.selection_probability,
    )
    filtered.to_csv(output / "latest_adjacency_stable.csv", index_label="source")

    edges = edge_list(filtered, tolerance=config.model.edge_tolerance)
    edges.to_csv(output / "edge_list_stable.csv", index=False)
    nodes = node_metrics(filtered, panel.metadata)
    nodes.to_csv(output / "node_metrics.csv", index=False)
    sectors = sector_spillovers(filtered, panel.metadata)
    sectors.to_csv(output / "sector_spillovers.csv", index=False)

    pairwise = fit_pairwise_delta_covar(
        latest_returns,
        latest_states,
        quantile=config.model.quantile,
        factor_count=config.model.factor_count,
        lag=0,
    )
    lagged_pairwise = fit_pairwise_delta_covar(
        latest_returns,
        latest_states,
        quantile=config.model.quantile,
        factor_count=config.model.factor_count,
        lag=config.model.lag,
    )
    tail_uplift = fit_tail_uplift_network(
        latest_returns, quantile=config.model.quantile, lag=config.model.lag
    )
    comparison = compare_networks(
        {
            "penalized_quantile": latest.adjacency,
            "contemporaneous_delta_covar": pairwise.adjacency,
            "lagged_pairwise_quantile": lagged_pairwise.adjacency,
            "lagged_tail_uplift": tail_uplift.adjacency,
        },
        top_k=min(config.output.top_edges, len(latest.adjacency) * (len(latest.adjacency) - 1)),
    )
    comparison.to_csv(output / "method_comparison.csv", index=False)

    recovery: dict[str, float | int] | None = None
    if panel.truth is not None:
        recovery = recovery_metrics(filtered, panel.truth, tolerance=config.model.edge_tolerance)
        _write_json(output / "simulation_recovery.json", recovery)

    plot_connectedness(history, figure_directory / "connectedness.png")
    plot_network(
        filtered,
        panel.metadata,
        figure_directory / "network_latest.png",
        top_edges=config.output.top_edges,
    )
    plot_sector_heatmap(sectors, figure_directory / "sector_heatmap.png")
    plot_stability(stability, list(panel.returns.columns), figure_directory / "edge_stability.png")

    manifest = {
        "software_version": __version__,
        "python_version": platform.python_version(),
        "quantile": config.model.quantile,
        "alpha": alpha,
        "factor_count": config.model.factor_count,
        "lag": config.model.lag,
        "window": config.model.window,
        "step": config.model.step,
        "rolling_windows": len(estimates),
        "bootstrap_reps": config.inference.bootstrap_reps,
        "selection_probability": config.inference.selection_probability,
        "input_sha256": {
            "returns": file_sha256(config.data.returns),
            "states": file_sha256(config.data.states) if config.data.states else None,
            "metadata": file_sha256(config.data.metadata),
            "truth": (
                file_sha256(config.data.truth)
                if config.data.truth is not None and config.data.truth.exists()
                else None
            ),
        },
    }
    _write_json(output / "run_manifest.json", manifest)
    _write_summary(
        output / "SUMMARY.md",
        manifest,
        history,
        connectedness_summary(filtered),
        nodes,
        recovery,
    )
    return {"manifest": manifest, "output_directory": output, "recovery": recovery}


def _write_summary(
    path: Path,
    manifest: dict[str, object],
    history: pd.DataFrame,
    stable_summary: dict[str, float | int],
    nodes: pd.DataFrame,
    recovery: dict[str, float | int] | None,
) -> None:
    latest = history.iloc[-1]
    top_emitters = nodes.head(5)[["institution", "out_strength", "net_transmitter"]]
    lines = [
        "# Synthetic demonstration results",
        "",
        "> These outputs use simulated data with a known contagion graph. They do not constitute",
        "> evidence about any real financial institution or the Chinese financial system.",
        "",
        "## Specification",
        "",
        f"- Lower-tail quantile: `{manifest['quantile']}`",
        f"- Fixed L1 penalty: `{manifest['alpha']}`",
        f"- Common factors removed: `{manifest['factor_count']}`",
        f"- Source-return lag: `{manifest['lag']}`",
        f"- Rolling windows: `{manifest['rolling_windows']}`",
        f"- Moving-block bootstrap repetitions: `{manifest['bootstrap_reps']}`",
        "",
        "## Latest network",
        "",
        f"- Unfiltered mean adverse edge weight: `{latest['mean_connectedness']:.6f}`",
        f"- Unfiltered active-edge density: `{latest['density']:.3f}`",
        f"- Stability-filtered active edges: `{stable_summary['active_edges']}`",
        f"- Stability-filtered density: `{stable_summary['density']:.3f}`",
        "",
        "Top emitters in the stability-filtered network:",
        "",
        top_emitters.to_markdown(index=False, floatfmt=".4f"),
    ]
    if recovery is not None:
        lines.extend(
            [
                "",
                "## Known-graph recovery diagnostic",
                "",
                f"- Precision: `{recovery['precision']:.3f}`",
                f"- Recall: `{recovery['recall']:.3f}`",
                f"- Average precision: `{recovery['average_precision']:.3f}`",
                "",
                "Recovery performance is a property of this stylized DGP and must not be read as",
                "a guarantee for observational financial data.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
