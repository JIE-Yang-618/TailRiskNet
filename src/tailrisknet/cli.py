"""Command-line entry points for simulation and complete research runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tailrisknet.config import ProjectConfig
from tailrisknet.pipeline import run_pipeline
from tailrisknet.simulation import simulate_financial_panel, write_simulated_panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tailrisknet", description="Tail-risk network research tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser(
        "simulate", help="generate a panel with a known contagion graph"
    )
    simulate.add_argument("--output", type=Path, default=Path("data/demo"))
    simulate.add_argument("--periods", type=int, default=520)
    simulate.add_argument("--institutions-per-sector", type=int, default=4)
    simulate.add_argument("--seed", type=int, default=2026)

    run = subparsers.add_parser("run", help="execute the configured end-to-end analysis")
    run.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        panel = simulate_financial_panel(
            periods=args.periods,
            institutions_per_sector=args.institutions_per_sector,
            random_seed=args.seed,
        )
        write_simulated_panel(panel, args.output)
        print(f"Wrote synthetic panel to {args.output.resolve()}")
        return
    if args.command == "run":
        result = run_pipeline(ProjectConfig.from_yaml(args.config))
        print(f"Analysis complete: {Path(str(result['output_directory'])).resolve()}")
        return
    raise RuntimeError(f"unknown command: {args.command}")
