from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable

from .config import RuntimeConfig
from .constants import ARCHITECTURE_ROLE_SEQUENCE, ARCHITECTURE_SYSTEM_PROMPTS, CHOICE_KEYS, DATASET_FILES
from .metrics import metric_bundle
from .parsing import extract_choice_and_reasoning, normalize_probabilities
from .prompts import build_prompt


def _role_label(role_name: str) -> str:
    return role_name.replace("_", " ").title()


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def _architecture_agent_count(architecture: str, num_iterations: int) -> int:
    base_count = len(ARCHITECTURE_ROLE_SEQUENCE[architecture])
    if architecture == "iteration":
        return base_count * max(1, num_iterations)
    return base_count


def _load_dataset(runtime_config: RuntimeConfig) -> list[dict[str, Any]]:
    dataset_filename = DATASET_FILES[runtime_config.experiment.dataset_type]
    dataset_path = runtime_config.paths.data_dir / dataset_filename
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        dataset_rows: list[dict[str, Any]] = json.load(dataset_file)

    start_index = max(0, runtime_config.experiment.start_index)
    sliced_rows = dataset_rows[start_index:]
    if runtime_config.experiment.max_questions is not None:
        sliced_rows = sliced_rows[: runtime_config.experiment.max_questions]
    return sliced_rows


def _build_output_directory(runtime_config: RuntimeConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = (
        f"{runtime_config.experiment.architecture}_"
        f"{runtime_config.experiment.dataset_type}_"
        f"{_sanitize_model_name(runtime_config.model.model_name)}_"
        f"{timestamp}"
    )
    output_directory = runtime_config.paths.output_dir / output_folder_name
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def _get_api_key(api_key_env: str) -> str:
    api_key_value = os.getenv(api_key_env)
    if api_key_value:
        return api_key_value
    print(f"[Warning] Environment variable '{api_key_env}' is not set. Using placeholder key 'EMPTY'.")
    return "EMPTY"


def _simulate_response(prompt_text: str, role_name: str) -> str:
    local_random = random.Random(f"{role_name}::{prompt_text}")
    raw_values = {
        "A": local_random.random() + 0.01,
        "B": local_random.random() + 0.01,
        "C": local_random.random() + 0.01,
    }
    normalized_values = normalize_probabilities(raw_values)
    response_payload = {
        "ChoiceProbabilities": normalized_values,
        "Reason": f"Dry-run output generated for role {role_name}.",
    }
    return "```json\n" + json.dumps(response_payload, ensure_ascii=False, indent=2) + "\n```"


def _query_model(
    client: Any,
    runtime_config: RuntimeConfig,
    system_prompt: str,
    user_prompt: str,
    role_label: str,
) -> tuple[str, float]:
    if runtime_config.experiment.dry_run:
        simulated_text = _simulate_response(user_prompt, role_label)
        return simulated_text, 0.0

    last_error: Exception | None = None
    for attempt_index in range(runtime_config.model.max_retries):
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model=runtime_config.model.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            elapsed_time = time.time() - start_time
            response_text = completion.choices[0].message.content or ""
            return response_text, elapsed_time
        except Exception as error:  # noqa: BLE001
            last_error = error
            should_retry = attempt_index + 1 < runtime_config.model.max_retries
            if not should_retry:
                break
            print(
                f"[Retry] API request failed ({attempt_index + 1}/{runtime_config.model.max_retries}). "
                f"Waiting {runtime_config.model.retry_wait_seconds}s. Error: {error}"
            )
            time.sleep(runtime_config.model.retry_wait_seconds)

    raise RuntimeError(f"API request failed after retries: {last_error}")


def _build_agent_record(
    agent_index: int,
    global_agent_index: int,
    iteration_index: int,
    role_name: str,
    system_prompt: str,
    prompt_text: str,
    response_text: str,
    parsed_output: dict[str, object],
    latency_seconds: float,
) -> dict[str, Any]:
    return {
        "agent_index": agent_index,
        "global_agent_index": global_agent_index,
        "iteration": iteration_index + 1,
        "agent_role": _role_label(role_name),
        "system_prompt": system_prompt,
        "prompt": prompt_text,
        "response": response_text,
        "choice_probabilities": parsed_output["choice_probabilities"],
        "reasoning": parsed_output["reasoning"],
        "latency_seconds": latency_seconds,
    }


def _run_non_iteration_question(
    client: Any,
    runtime_config: RuntimeConfig,
    templates: list[str],
) -> list[dict[str, Any]]:
    architecture = runtime_config.experiment.architecture
    role_sequence = ARCHITECTURE_ROLE_SEQUENCE[architecture]
    system_prompts = ARCHITECTURE_SYSTEM_PROMPTS[architecture]

    completed_agents: list[dict[str, Any]] = []
    for agent_index, role_name in enumerate(role_sequence):
        prompt_text = build_prompt(
            architecture=architecture,
            agent_index=agent_index,
            templates=templates,
            completed_agents=completed_agents,
        )
        response_text, latency_seconds = _query_model(
            client=client,
            runtime_config=runtime_config,
            system_prompt=system_prompts[agent_index],
            user_prompt=prompt_text,
            role_label=_role_label(role_name),
        )
        parsed_output = extract_choice_and_reasoning(response_text)
        agent_record = _build_agent_record(
            agent_index=agent_index,
            global_agent_index=agent_index,
            iteration_index=0,
            role_name=role_name,
            system_prompt=system_prompts[agent_index],
            prompt_text=prompt_text,
            response_text=response_text,
            parsed_output=parsed_output,
            latency_seconds=latency_seconds,
        )
        completed_agents.append(agent_record)
    return completed_agents


def _run_iteration_question(
    client: Any,
    runtime_config: RuntimeConfig,
    templates: list[str],
) -> list[dict[str, Any]]:
    architecture = runtime_config.experiment.architecture
    role_sequence = ARCHITECTURE_ROLE_SEQUENCE[architecture]
    system_prompts = ARCHITECTURE_SYSTEM_PROMPTS[architecture]

    all_iteration_agents: list[dict[str, Any]] = []
    previous_iteration_summary: str | None = None
    for iteration_index in range(runtime_config.experiment.num_iterations):
        completed_iteration_agents: list[dict[str, Any]] = []
        for agent_index, role_name in enumerate(role_sequence):
            prompt_text = build_prompt(
                architecture=architecture,
                agent_index=agent_index,
                templates=templates,
                completed_agents=completed_iteration_agents,
                iteration_index=iteration_index,
                previous_iteration_summary=previous_iteration_summary,
            )
            response_text, latency_seconds = _query_model(
                client=client,
                runtime_config=runtime_config,
                system_prompt=system_prompts[agent_index],
                user_prompt=prompt_text,
                role_label=_role_label(role_name),
            )
            parsed_output = extract_choice_and_reasoning(response_text)

            global_agent_index = iteration_index * len(role_sequence) + agent_index
            agent_record = _build_agent_record(
                agent_index=agent_index,
                global_agent_index=global_agent_index,
                iteration_index=iteration_index,
                role_name=role_name,
                system_prompt=system_prompts[agent_index],
                prompt_text=prompt_text,
                response_text=response_text,
                parsed_output=parsed_output,
                latency_seconds=latency_seconds,
            )
            completed_iteration_agents.append(agent_record)
            all_iteration_agents.append(agent_record)

        previous_iteration_summary = str(completed_iteration_agents[-1]["reasoning"])

    return all_iteration_agents


def _write_csv(file_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        file_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with file_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _choice_value(choice_probabilities: dict[str, float], choice_key: str) -> float:
    return float(choice_probabilities.get(choice_key, 0.0))


def _build_response_rows(all_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response_rows: list[dict[str, Any]] = []
    for question_record in all_responses:
        templates = question_record.get("templates", [])
        demographics = question_record.get("demographic_info", [])
        final_choice = question_record.get("final_choice_probabilities", {})

        for agent_record in question_record.get("agents", []):
            choice_probabilities = agent_record["choice_probabilities"]
            row: dict[str, Any] = {
                "question_id": question_record["question_id"],
                "iteration": agent_record["iteration"],
                "agent_index": agent_record["agent_index"] + 1,
                "global_agent_index": agent_record["global_agent_index"] + 1,
                "agent_role": agent_record["agent_role"],
                "agent_choice_A_prob": _choice_value(choice_probabilities, "A"),
                "agent_choice_B_prob": _choice_value(choice_probabilities, "B"),
                "agent_choice_C_prob": _choice_value(choice_probabilities, "C"),
                "agent_reasoning": agent_record["reasoning"],
                "agent_prompt": agent_record["prompt"],
                "agent_response": agent_record["response"],
                "latency_seconds": agent_record["latency_seconds"],
                "final_choice_A_prob": _choice_value(final_choice, "A"),
                "final_choice_B_prob": _choice_value(final_choice, "B"),
                "final_choice_C_prob": _choice_value(final_choice, "C"),
            }

            for template_index, choice_key in enumerate(CHOICE_KEYS):
                template_text = templates[template_index] if template_index < len(templates) else ""
                demographic_row = demographics[template_index] if template_index < len(demographics) else {}
                row[f"template_{choice_key}"] = template_text
                row[f"demographic_{choice_key}_age"] = demographic_row.get("age", "")
                row[f"demographic_{choice_key}_gender"] = demographic_row.get("gender", "")
                row[f"demographic_{choice_key}_race"] = demographic_row.get("race", "")

            response_rows.append(row)

    return response_rows


def _build_metric_rows(all_responses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    question_metric_rows: list[dict[str, Any]] = []

    aggregate_sums: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "choice_A_prob": 0.0,
            "choice_B_prob": 0.0,
            "choice_C_prob": 0.0,
            "variance": 0.0,
            "entropy": 0.0,
            "gini": 0.0,
            "uniform_kl": 0.0,
        }
    )
    aggregate_counts: dict[int, int] = defaultdict(int)
    aggregate_meta: dict[int, dict[str, Any]] = {}

    for question_record in all_responses:
        for agent_record in question_record["agents"]:
            choice_probabilities = agent_record["choice_probabilities"]
            metric_values = metric_bundle(choice_probabilities)

            row = {
                "question_id": question_record["question_id"],
                "iteration": agent_record["iteration"],
                "agent_index": agent_record["agent_index"] + 1,
                "global_agent_index": agent_record["global_agent_index"] + 1,
                "agent_role": agent_record["agent_role"],
                "choice_A_prob": _choice_value(choice_probabilities, "A"),
                "choice_B_prob": _choice_value(choice_probabilities, "B"),
                "choice_C_prob": _choice_value(choice_probabilities, "C"),
                "variance": metric_values["variance"],
                "entropy": metric_values["entropy"],
                "gini": metric_values["gini"],
                "uniform_kl": metric_values["uniform_kl"],
            }
            question_metric_rows.append(row)

            key = agent_record["global_agent_index"]
            aggregate_meta[key] = {
                "iteration": agent_record["iteration"],
                "agent_index": agent_record["agent_index"] + 1,
                "global_agent_index": agent_record["global_agent_index"] + 1,
                "agent_role": agent_record["agent_role"],
            }
            aggregate_counts[key] += 1
            aggregate_sums[key]["choice_A_prob"] += row["choice_A_prob"]
            aggregate_sums[key]["choice_B_prob"] += row["choice_B_prob"]
            aggregate_sums[key]["choice_C_prob"] += row["choice_C_prob"]
            aggregate_sums[key]["variance"] += row["variance"]
            aggregate_sums[key]["entropy"] += row["entropy"]
            aggregate_sums[key]["gini"] += row["gini"]
            aggregate_sums[key]["uniform_kl"] += row["uniform_kl"]

    average_rows: list[dict[str, Any]] = []
    for key in sorted(aggregate_counts):
        count_value = aggregate_counts[key]
        sums = aggregate_sums[key]
        meta = aggregate_meta[key]
        average_rows.append(
            {
                **meta,
                "questions_processed": count_value,
                "avg_choice_A_prob": sums["choice_A_prob"] / count_value,
                "avg_choice_B_prob": sums["choice_B_prob"] / count_value,
                "avg_choice_C_prob": sums["choice_C_prob"] / count_value,
                "avg_variance": sums["variance"] / count_value,
                "avg_entropy": sums["entropy"] / count_value,
                "avg_gini": sums["gini"] / count_value,
                "avg_uniform_kl": sums["uniform_kl"] / count_value,
            }
        )

    return question_metric_rows, average_rows


def _save_progress(all_responses: list[dict[str, Any]], runtime_config: RuntimeConfig, output_directory: Path) -> None:
    processed_questions = len(all_responses)
    prefix = (
        f"{_sanitize_model_name(runtime_config.model.model_name)}_"
        f"{runtime_config.experiment.dataset_type}_"
        f"{runtime_config.experiment.architecture}"
    )

    metric_rows, average_rows = _build_metric_rows(all_responses)
    response_rows = _build_response_rows(all_responses)

    metrics_path = output_directory / f"{prefix}_question_metrics_progress_{processed_questions}.csv"
    averages_path = output_directory / f"{prefix}_avg_metrics_progress_{processed_questions}.csv"
    responses_csv_path = output_directory / f"{prefix}_responses_progress_{processed_questions}.csv"
    responses_json_path = output_directory / f"{prefix}_responses_progress_{processed_questions}.json"

    _write_csv(metrics_path, metric_rows)
    _write_csv(averages_path, average_rows)
    _write_csv(responses_csv_path, response_rows)
    with responses_json_path.open("w", encoding="utf-8") as responses_json_file:
        json.dump(all_responses, responses_json_file, ensure_ascii=False, indent=2)

    print(f"[Saved] {metrics_path}")
    print(f"[Saved] {averages_path}")
    print(f"[Saved] {responses_csv_path}")
    print(f"[Saved] {responses_json_path}")


def run_experiment(runtime_config: RuntimeConfig) -> Path:
    random.seed(runtime_config.experiment.seed)

    dataset_rows = _load_dataset(runtime_config)
    if not dataset_rows:
        raise RuntimeError("No questions found after applying dataset slicing options.")

    output_directory = _build_output_directory(runtime_config)
    output_directory.mkdir(parents=True, exist_ok=True)

    run_config_path = output_directory / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as run_config_file:
        json.dump(runtime_config.to_dict(), run_config_file, ensure_ascii=False, indent=2)

    client: Any = None
    if not runtime_config.experiment.dry_run:
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency 'openai'. Install dependencies with `pip install -r requirements.txt`."
            ) from error

        client = OpenAI(
            api_key=_get_api_key(runtime_config.model.api_key_env),
            base_url=runtime_config.model.base_url,
            timeout=runtime_config.model.timeout_seconds,
        )

    all_responses: list[dict[str, Any]] = []
    architecture = runtime_config.experiment.architecture
    total_positions = _architecture_agent_count(architecture, runtime_config.experiment.num_iterations)
    agent_choice_totals = [{choice_key: 0.0 for choice_key in CHOICE_KEYS} for _ in range(total_positions)]

    print(f"[Run] Output directory: {output_directory}")
    print(f"[Run] Total questions: {len(dataset_rows)}")
    print(f"[Run] Architecture: {architecture}")
    print(f"[Run] Model: {runtime_config.model.model_name}")
    print(f"[Run] Dry run: {runtime_config.experiment.dry_run}")

    for question_record in tqdm(dataset_rows, desc="Running questions"):
        question_id = question_record["question_id"]
        templates = question_record["templates"]
        demographics = question_record.get("demographic_info", [])

        if architecture == "iteration":
            agent_records = _run_iteration_question(
                client=client,
                runtime_config=runtime_config,
                templates=templates,
            )
        else:
            agent_records = _run_non_iteration_question(
                client=client,
                runtime_config=runtime_config,
                templates=templates,
            )

        for agent_record in agent_records:
            global_agent_index = agent_record["global_agent_index"]
            for choice_key in CHOICE_KEYS:
                probability_value = float(agent_record["choice_probabilities"][choice_key])
                agent_choice_totals[global_agent_index][choice_key] += probability_value

        final_choice_probabilities = agent_records[-1]["choice_probabilities"]
        all_responses.append(
            {
                "question_id": question_id,
                "templates": templates,
                "demographic_info": demographics,
                "agents": agent_records,
                "final_choice_probabilities": final_choice_probabilities,
            }
        )

        print(
            f"[Question {question_id}] Final choice probs: "
            f"A={final_choice_probabilities['A']:.1f}, "
            f"B={final_choice_probabilities['B']:.1f}, "
            f"C={final_choice_probabilities['C']:.1f}"
        )

        should_save_interval = len(all_responses) % runtime_config.experiment.save_interval == 0
        is_last_question = len(all_responses) == len(dataset_rows)
        if should_save_interval or is_last_question:
            _save_progress(all_responses, runtime_config, output_directory)

    print("[Summary] Mean probabilities per agent position:")
    processed_questions = len(all_responses)
    for position_index, choice_totals in enumerate(agent_choice_totals, start=1):
        print(
            f"  Agent {position_index}: "
            f"A={choice_totals['A'] / processed_questions:.3f}, "
            f"B={choice_totals['B'] / processed_questions:.3f}, "
            f"C={choice_totals['C'] / processed_questions:.3f}"
        )

    print(f"[Done] Results saved under {output_directory}")
    return output_directory
