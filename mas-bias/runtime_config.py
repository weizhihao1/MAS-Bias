import json
import os
from copy import deepcopy

from openai import OpenAI

_DATASET_FILE_MAP = {
    "explicit": "explicit_prompts.json",
    "implicit": "implicit_prompts.json",
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_runtime_config(default_model: str, default_dataset_type: str = "implicit") -> dict:
    dataset_type = os.getenv("DATASET_TYPE", default_dataset_type).strip().lower()
    if dataset_type not in _DATASET_FILE_MAP:
        valid = ", ".join(sorted(_DATASET_FILE_MAP))
        raise ValueError(f"Unsupported DATASET_TYPE={dataset_type!r}. Valid values: {valid}.")

    model_name = os.getenv("MODEL_NAME", default_model).strip() or default_model
    dataset_path = os.path.join(_project_root(), "data", _DATASET_FILE_MAP[dataset_type])

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    return {
        "model_name": model_name,
        "dataset_type": dataset_type,
        "dataset_path": dataset_path,
        "api_key_env": os.getenv("API_KEY_ENV", "OPENAI_API_KEY"),
        "base_url": os.getenv("BASE_URL", "").strip(),
    }


def build_openai_client(runtime_config: dict) -> OpenAI:
    api_key_env = runtime_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env, "").strip()

    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Please export the environment variable {api_key_env}."
        )

    client_kwargs = {"api_key": api_key}
    base_url = runtime_config.get("base_url", "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def resolve_agent_models(default_models: dict) -> dict:
    raw = os.getenv("MIXED_AGENT_MODELS_JSON", "").strip()
    if not raw:
        return deepcopy(default_models)

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("MIXED_AGENT_MODELS_JSON must be a JSON object mapping index to model.")

    merged = deepcopy(default_models)
    for agent_index, model_name in parsed.items():
        agent_index_int = int(agent_index)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"Invalid model name for agent {agent_index}: {model_name!r}")
        merged[agent_index_int] = model_name.strip()

    return merged
