#!/usr/bin/env bash
# Launch TensorRT-LLM with flags from the matrix config.
# Writes configs/.active_engine before starting the server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../scripts/load_env.sh
source "$ROOT/scripts/load_env.sh"

if [[ ! -f "$MATRIX" ]]; then
  echo "Missing matrix config: $MATRIX" >&2
  exit 1
fi

eval "$(python3 "$ROOT/scripts/engine_config.py" serve-vars tensorrt_llm "$MATRIX")"

echo "Starting TensorRT-LLM from $MATRIX"
echo "  model=$MODEL host=$HOST port=$PORT tp=$TP max_len=$MAX_LEN"

exec trtllm-serve serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --max_seq_len "$MAX_LEN" \
  $EXTRA_ARGS
