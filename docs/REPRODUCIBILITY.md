# Reproducibility protocol

## Deterministic demonstration

```bash
python -m pip install -e ".[dev]"
tailrisknet simulate --output data/demo --periods 520 --institutions-per-sector 4 --seed 2026
tailrisknet run --config configs/demo.yaml
pytest
```

Simulation, network layout, cross-validation splits, and bootstrap draws all use explicit seeds. The run manifest records the package version, Python version, complete estimation specification, and SHA-256 hashes of every input file.

## Research-run checklist

Before interpreting an empirical run:

1. verify returns, corporate actions, listing dates, and weekly alignment;
2. document whether prices are adjusted and whether returns are simple or logarithmic;
3. lag state variables when their publication timing requires it;
4. choose the rolling window and quantile before inspecting crisis-period rankings;
5. calibrate the penalty only on the designated initial window;
6. compare raw and common-factor-adjusted networks;
7. report unfiltered and stability-filtered results;
8. report sector means alongside sector totals;
9. retain negative/beneficial signed responses even though the adverse graph truncates them;
10. avoid causal language unless identification comes from outside the network regression.

## Output audit trail

Each run produces:

- `run_manifest.json`: model specification and input hashes;
- `connectedness.csv`: rolling aggregate summaries;
- `latest_adjacency.csv`: unfiltered adverse network;
- `latest_signed_effects.csv`: signed stress responses;
- `edge_stability.csv`: bootstrap diagnostics;
- `latest_adjacency_stable.csv`: thresholded reporting network;
- `edge_list_stable.csv`: sorted stable directed edges;
- `node_metrics.csv`: institution rankings;
- `sector_spillovers.csv`: total and normalized sector blocks;
- `method_comparison.csv`: alternative-estimator agreement;
- figures and a concise Markdown summary.

The curated `results/demo/` directory is intentionally versioned so that visitors can inspect an end-to-end research product without proprietary inputs.

