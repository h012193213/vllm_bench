#!/usr/bin/env bash
# Orchestrate fixed-criteria GuideLLM benchmarks across scenarios.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=load_env.sh
source "$ROOT/scripts/load_env.sh"
exec python3 "$ROOT/scripts/run_benchmark.py" "$@"
