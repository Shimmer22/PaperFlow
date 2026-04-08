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

if [[ -z "${API_KEY:-}" && -z "${NV_API_KEY:-}" ]]; then
  echo "No API key found. Put API_KEY=... or NV_API_KEY=... in .env or env." >&2
  exit 1
fi

cd ui
if [[ ! -d "node_modules" || ! -x "node_modules/.bin/electron" ]]; then
  echo "UI dependencies not found, running npm install..."
  npm install
fi
npm start
