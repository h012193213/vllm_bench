#!/usr/bin/env bash
# Load repo .env and resolve paths. Source from other scripts:
#   source "$ROOT/scripts/load_env.sh"

if [[ -z "${VLLM_BENCH_ROOT:-}" ]]; then
  VLLM_BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -f "$VLLM_BENCH_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$VLLM_BENCH_ROOT/.env"
  set +a
fi

if [[ -n "${MATRIX:-}" && "${MATRIX}" != /* ]]; then
  MATRIX="$VLLM_BENCH_ROOT/$MATRIX"
fi
MATRIX="${MATRIX:-$VLLM_BENCH_ROOT/configs/benchmark_matrix.yaml}"

export VLLM_BENCH_ROOT MATRIX
