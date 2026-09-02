"""Configuration loading with explicit validation and path resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    returns: Path
    states: Path | None
    metadata: Path
    truth: Path | None = None


@dataclass(frozen=True)
class ModelConfig:
    quantile: float = 0.05
    window: int = 156
    step: int = 13
    min_observations: int = 120
    alpha: float | None = 0.01
    alpha_grid: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
    factor_count: int = 1
    lag: int = 1
    edge_tolerance: float = 1e-6


@dataclass(frozen=True)
class InferenceConfig:
    bootstrap_reps: int = 100
    block_length: int = 8
    selection_probability: float = 0.70
    random_seed: int = 2026


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    top_edges: int = 30


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    model: ModelConfig
    inference: InferenceConfig
    output: OutputConfig
    source_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectConfig:
        source_path = Path(path).expanduser().resolve()
        with source_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a YAML mapping")

        base = source_path.parent

        def resolve(value: str | None) -> Path | None:
            if value is None:
                return None
            candidate = Path(value).expanduser()
            return (base / candidate).resolve() if not candidate.is_absolute() else candidate

        data_raw = raw.get("data", {})
        model_raw = raw.get("model", {})
        inference_raw = raw.get("inference", {})
        output_raw = raw.get("output", {})

        config = cls(
            data=DataConfig(
                returns=_required_path(data_raw, "returns", resolve),
                states=resolve(data_raw.get("states")),
                metadata=_required_path(data_raw, "metadata", resolve),
                truth=resolve(data_raw.get("truth")),
            ),
            model=ModelConfig(
                quantile=float(model_raw.get("quantile", 0.05)),
                window=int(model_raw.get("window", 156)),
                step=int(model_raw.get("step", 13)),
                min_observations=int(model_raw.get("min_observations", 120)),
                alpha=(None if model_raw.get("alpha") is None else float(model_raw["alpha"])),
                alpha_grid=tuple(
                    float(value)
                    for value in model_raw.get("alpha_grid", [0.0025, 0.005, 0.01, 0.02])
                ),
                factor_count=int(model_raw.get("factor_count", 1)),
                lag=int(model_raw.get("lag", 1)),
                edge_tolerance=float(model_raw.get("edge_tolerance", 1e-6)),
            ),
            inference=InferenceConfig(
                bootstrap_reps=int(inference_raw.get("bootstrap_reps", 100)),
                block_length=int(inference_raw.get("block_length", 8)),
                selection_probability=float(inference_raw.get("selection_probability", 0.70)),
                random_seed=int(inference_raw.get("random_seed", 2026)),
            ),
            output=OutputConfig(
                directory=_required_path(output_raw, "directory", resolve),
                top_edges=int(output_raw.get("top_edges", 30)),
            ),
            source_path=source_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0 < self.model.quantile < 0.5:
            raise ValueError("quantile must lie strictly between 0 and 0.5")
        if self.model.window < self.model.min_observations:
            raise ValueError("window must be at least min_observations")
        if self.model.step < 1:
            raise ValueError("step must be positive")
        if self.model.factor_count < 0:
            raise ValueError("factor_count cannot be negative")
        if self.model.lag < 0:
            raise ValueError("lag cannot be negative")
        if self.model.alpha is not None and self.model.alpha < 0:
            raise ValueError("alpha cannot be negative")
        if not self.model.alpha_grid or min(self.model.alpha_grid) < 0:
            raise ValueError("alpha_grid must contain non-negative values")
        if self.inference.bootstrap_reps < 0:
            raise ValueError("bootstrap_reps cannot be negative")
        if self.inference.block_length < 1:
            raise ValueError("block_length must be positive")
        if not 0 <= self.inference.selection_probability <= 1:
            raise ValueError("selection_probability must lie in [0, 1]")

    def as_serializable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _stringify_paths(payload)


def _required_path(
    mapping: dict[str, Any], key: str, resolver: Any
) -> Path:  # resolver is local by design
    if key not in mapping:
        raise ValueError(f"missing required configuration key: {key}")
    path = resolver(mapping[key])
    if path is None:
        raise ValueError(f"configuration path cannot be null: {key}")
    return path


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value
