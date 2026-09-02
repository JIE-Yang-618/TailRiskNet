from __future__ import annotations

from pathlib import Path

from tailrisknet.config import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    OutputConfig,
    ProjectConfig,
)
from tailrisknet.pipeline import run_pipeline
from tailrisknet.simulation import simulate_financial_panel, write_simulated_panel


def test_end_to_end_pipeline_writes_auditable_outputs(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    output_directory = tmp_path / "results"
    panel = simulate_financial_panel(periods=160, institutions_per_sector=2, random_seed=15)
    write_simulated_panel(panel, data_directory)
    config = ProjectConfig(
        data=DataConfig(
            returns=data_directory / "returns.csv",
            states=data_directory / "states.csv",
            metadata=data_directory / "institutions.csv",
            truth=data_directory / "true_edges.csv",
        ),
        model=ModelConfig(
            quantile=0.05,
            window=120,
            step=40,
            min_observations=100,
            alpha=0.01,
            factor_count=1,
        ),
        inference=InferenceConfig(
            bootstrap_reps=3,
            block_length=6,
            selection_probability=0.50,
            random_seed=15,
        ),
        output=OutputConfig(directory=output_directory, top_edges=12),
        source_path=tmp_path / "test.yaml",
    )

    result = run_pipeline(config)

    assert result["output_directory"] == output_directory
    for filename in (
        "run_manifest.json",
        "connectedness.csv",
        "latest_adjacency_stable.csv",
        "edge_stability.csv",
        "method_comparison.csv",
        "simulation_recovery.json",
        "SUMMARY.md",
    ):
        assert (output_directory / filename).exists()
    assert (output_directory / "figures" / "network_latest.png").exists()
