#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$ROOT_DIR/configs"

print_usage() {
  cat <<'EOF'
Usage:
  ./run_experiment.sh --list
  ./run_experiment.sh <config_name> [KEY=VALUE ...]

Examples:
  ./run_experiment.sh linear_plain
  ./run_experiment.sh linear_persona DATASET_TYPE=explicit MODEL_NAME=gpt-4o
  ./run_experiment.sh different_model MIXED_AGENT_MODELS_JSON='{"0":"gpt-4o-mini","5":"deepseek-r1"}'

Notes:
  - Put API key in the env var specified by API_KEY_ENV (default: OPENAI_API_KEY).
  - Optional KEY=VALUE overrides are applied after loading configs/<config_name>.env.
EOF
}

if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

if [[ "${1:-}" == "--list" ]]; then
  echo "Available configs:"
  for f in "$CONFIG_DIR"/*.env; do
    basename "$f" .env
  done | sort
  exit 0
fi

CONFIG_NAME="$1"
shift
CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config not found: $CONFIG_FILE"
  echo
  print_usage
  exit 1
fi

set -a
source "$CONFIG_FILE"
set +a

for override in "$@"; do
  if [[ "$override" != *=* ]]; then
    echo "Invalid override: $override"
    echo "Use KEY=VALUE format."
    exit 1
  fi
  export "$override"
done

: "${SCRIPT_NAME:?SCRIPT_NAME is required in config file.}"
SCRIPT_PATH="$ROOT_DIR/mas-bias/$SCRIPT_NAME"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Python entry script not found: $SCRIPT_PATH"
  exit 1
fi

API_ENV_NAME="${API_KEY_ENV:-OPENAI_API_KEY}"
if [[ -z "${!API_ENV_NAME:-}" ]]; then
  echo "Missing API key env var: $API_ENV_NAME"
  echo "Example: export $API_ENV_NAME=your_api_key"
  exit 1
fi

echo "[MAS-Bias] Config      : $CONFIG_NAME"
echo "[MAS-Bias] Script      : $SCRIPT_NAME"
echo "[MAS-Bias] Dataset     : ${DATASET_TYPE:-implicit}"
echo "[MAS-Bias] Model       : ${MODEL_NAME:-mixed-agent-config}"
echo "[MAS-Bias] Results dir : ${RESULT_DIR:-<defined by script>}"

python "$SCRIPT_PATH"
