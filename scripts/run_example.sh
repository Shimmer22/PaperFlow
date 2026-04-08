#!/usr/bin/env bash
set -euo pipefail

python -m research_flow.cli run \
  --idea-file examples/idea_example.md \
  --config config.example.yaml \
  --provider-config providers/codex.example.yaml \
  --outdir outputs/example_run

