#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

shopt -s nullglob
matches=(research-flow-provider-*)

if (( ${#matches[@]} == 0 )); then
  echo "No research-flow-provider-* directories found in $ROOT_DIR"
  exit 0
fi

for d in "${matches[@]}"; do
  if [[ -d "$d" ]]; then
    rm -rf -- "$d"
    echo "removed $d"
  fi
done
