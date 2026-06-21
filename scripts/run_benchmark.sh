#!/usr/bin/env bash
# Orchestrate fixed-criteria GuideLLM benchmarks across scenarios.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/run_benchmark.py" "$@"
