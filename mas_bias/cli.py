from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_runtime_config
from .constants import ARCHITECTURE_CHOICES


def _set_override(overrides: dict[str, Any], key_path: list[str], value: Any) -> None:
    if value is None:
        return
    current_node = overrides
    for key_name in key_path[:-1]:
        if key_name not in current_node:
            current_node[key_name] = {}
        current_node = current_node[key_name]
    current_node[key_path[-1]] = value


def _build_overrides(parsed_args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    _set_override(overrides, ["experiment", "architecture"], parsed_args.architecture)
    _set_override(overrides, ["experiment", "dataset_type"], parsed_args.dataset_type)
    _set_override(overrides, ["experiment", "save_interval"], parsed_args.save_interval)
    _set_override(overrides, ["experiment", "num_iterations"], parsed_args.num_iterations)
    _set_override(overrides, ["experiment", "max_questions"], parsed_args.max_questions)
    _set_override(overrides, ["experiment", "start_index"], parsed_args.start_index)
    _set_override(overrides, ["experiment", "seed"], parsed_args.seed)
    _set_override(overrides, ["experiment", "dry_run"], parsed_args.dry_run)

    _set_override(overrides, ["model", "model_name"], parsed_args.model_name)
    _set_override(overrides, ["model", "api_key_env"], parsed_args.api_key_env)
    _set_override(overrides, ["model", "base_url"], parsed_args.base_url)
    _set_override(overrides, ["model", "max_retries"], parsed_args.max_retries)
    _set_override(overrides, ["model", "retry_wait_seconds"], parsed_args.retry_wait_seconds)
    _set_override(overrides, ["model", "timeout_seconds"], parsed_args.timeout_seconds)

    _set_override(overrides, ["paths", "data_dir"], parsed_args.data_dir)
    _set_override(overrides, ["paths", "output_dir"], parsed_args.output_dir)

    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MAS-Bias experiments with configurable architecture and model settings.")

    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to a JSON config file.")

    parser.add_argument("--architecture", type=str, choices=ARCHITECTURE_CHOICES, help="Experiment architecture to run.")
    parser.add_argument("--dataset-type", type=str, choices=["explicit", "implicit"], help="Dataset split to evaluate.")
    parser.add_argument("--save-interval", type=int, help="Save checkpoint files every N processed questions.")
    parser.add_argument("--num-iterations", type=int, help="Number of repeated units for the iteration architecture.")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to run.")
    parser.add_argument("--start-index", type=int, help="Starting index inside the dataset.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Skip API calls and generate deterministic mock outputs.")

    parser.add_argument("--model-name", type=str, help="Model name sent to the OpenAI-compatible API.")
    parser.add_argument("--api-key-env", type=str, help="Environment variable name containing API key.")
    parser.add_argument("--base-url", type=str, help="OpenAI-compatible API base URL.")
    parser.add_argument("--max-retries", type=int, help="Maximum retry attempts for failed requests.")
    parser.add_argument("--retry-wait-seconds", type=float, help="Retry wait interval in seconds.")
    parser.add_argument("--timeout-seconds", type=int, help="HTTP timeout in seconds.")

    parser.add_argument("--data-dir", type=str, help="Directory containing prompt datasets.")
    parser.add_argument("--output-dir", type=str, help="Directory used to store experiment outputs.")

    return parser


def main() -> None:
    parser = _build_parser()
    parsed_args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(parsed_args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    overrides = _build_overrides(parsed_args)
    runtime_config = load_runtime_config(config_path=config_path, overrides=overrides, project_root=project_root)

    from .runner import run_experiment

    run_experiment(runtime_config)


if __name__ == "__main__":
    main()
