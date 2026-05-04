#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${DS_API_KEY:-}" ]]; then
  echo "DS_API_KEY is missing. Put DS_API_KEY=... in .env or env." >&2
  exit 1
fi

PYTHON_BIN="./.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" -m research_flow.cli run \
  --provider deepseek \
  --provider-config providers/deepseek.example.yaml \
  "$@"
