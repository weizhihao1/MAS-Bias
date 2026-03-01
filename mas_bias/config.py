from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import ARCHITECTURE_CHOICES, DATASET_FILES


@dataclass
class ModelConfig:
    model_name: str
    api_key_env: str
    base_url: str
    max_retries: int
    retry_wait_seconds: float
    timeout_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "retry_wait_seconds": self.retry_wait_seconds,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ExperimentConfig:
    architecture: str
    dataset_type: str
    save_interval: int
    num_iterations: int
    max_questions: int | None
    start_index: int
    seed: int
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "dataset_type": self.dataset_type,
            "save_interval": self.save_interval,
            "num_iterations": self.num_iterations,
            "max_questions": self.max_questions,
            "start_index": self.start_index,
            "seed": self.seed,
            "dry_run": self.dry_run,
        }


@dataclass
class PathConfig:
    data_dir: Path
    output_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
        }


@dataclass
class RuntimeConfig:
    model: ModelConfig
    experiment: ExperimentConfig
    paths: PathConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "experiment": self.experiment.to_dict(),
            "paths": self.paths.to_dict(),
        }


def _deep_merge(base_value: dict[str, Any], override_value: dict[str, Any]) -> dict[str, Any]:
    merged_value = dict(base_value)
    for key, value in override_value.items():
        if isinstance(value, dict) and isinstance(merged_value.get(key), dict):
            merged_value[key] = _deep_merge(merged_value[key], value)
        else:
            merged_value[key] = value
    return merged_value


def _resolve_path(project_root: Path, path_text: str) -> Path:
    path_object = Path(path_text)
    if path_object.is_absolute():
        return path_object
    return (project_root / path_object).resolve()


def _validate(raw_config: dict[str, Any]) -> None:
    architecture_name = raw_config["experiment"]["architecture"]
    dataset_type = raw_config["experiment"]["dataset_type"]
    if architecture_name not in ARCHITECTURE_CHOICES:
        valid_architectures = ", ".join(ARCHITECTURE_CHOICES)
        raise ValueError(f"Unsupported architecture '{architecture_name}'. Valid options: {valid_architectures}")
    if dataset_type not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset type '{dataset_type}'. Use one of: {', '.join(DATASET_FILES)}")


def load_runtime_config(config_path: Path, overrides: dict[str, Any], project_root: Path) -> RuntimeConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        base_config: dict[str, Any] = json.load(config_file)

    merged_config = _deep_merge(base_config, overrides)
    _validate(merged_config)

    model_config = ModelConfig(
        model_name=merged_config["model"]["model_name"],
        api_key_env=merged_config["model"]["api_key_env"],
        base_url=merged_config["model"]["base_url"],
        max_retries=int(merged_config["model"]["max_retries"]),
        retry_wait_seconds=float(merged_config["model"]["retry_wait_seconds"]),
        timeout_seconds=int(merged_config["model"]["timeout_seconds"]),
    )

    max_questions_value = merged_config["experiment"].get("max_questions")
    if max_questions_value is not None:
        max_questions_value = int(max_questions_value)

    experiment_config = ExperimentConfig(
        architecture=merged_config["experiment"]["architecture"],
        dataset_type=merged_config["experiment"]["dataset_type"],
        save_interval=int(merged_config["experiment"]["save_interval"]),
        num_iterations=int(merged_config["experiment"]["num_iterations"]),
        max_questions=max_questions_value,
        start_index=int(merged_config["experiment"]["start_index"]),
        seed=int(merged_config["experiment"]["seed"]),
        dry_run=bool(merged_config["experiment"]["dry_run"]),
    )

    path_config = PathConfig(
        data_dir=_resolve_path(project_root, merged_config["paths"]["data_dir"]),
        output_dir=_resolve_path(project_root, merged_config["paths"]["output_dir"]),
    )

    return RuntimeConfig(model=model_config, experiment=experiment_config, paths=path_config)

