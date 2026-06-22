#!/usr/bin/env bash
# Launch SGLang with flags from the matrix config.
# Writes configs/.active_engine before starting the server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../scripts/load_env.sh
source "$ROOT/scripts/load_env.sh"

if [[ ! -f "$MATRIX" ]]; then
  echo "Missing matrix config: $MATRIX" >&2
  exit 1
fi

eval "$(python3 "$ROOT/scripts/engine_config.py" serve-vars sglang "$MATRIX")"

echo "Starting SGLang from $MATRIX"
echo "  model=$MODEL host=$HOST port=$PORT tp=$TP max_len=$MAX_LEN"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --context-length "$MAX_LEN" \
  --mem-fraction-static "$MEM_FRAC" \
  $EXTRA_ARGS
