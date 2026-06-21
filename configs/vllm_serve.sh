#!/usr/bin/env bash
# Launch vLLM with flags frozen in configs/benchmark_matrix.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="${ROOT}/configs/benchmark_matrix.yaml"

if [[ ! -f "$MATRIX" ]]; then
  echo "Missing matrix config: $MATRIX" >&2
  exit 1
fi

eval "$(python3 - "$MATRIX" <<'PY'
import shlex
import sys
import yaml
from pathlib import Path

matrix = yaml.safe_load(Path(sys.argv[1]).read_text())
model = matrix["model"]
vllm = matrix["vllm"]
print(f"MODEL={shlex.quote(model)}")
print(f"HOST={shlex.quote(str(vllm.get('host', '0.0.0.0')))}")
print(f"PORT={shlex.quote(str(vllm.get('port', 8000)))}")
print(f"TP={shlex.quote(str(vllm.get('tensor_parallel_size', 1)))}")
print(f"GPU_MEM={shlex.quote(str(vllm.get('gpu_memory_utilization', 0.90)))}")
print(f"MAX_LEN={shlex.quote(str(vllm.get('max_model_len', 8192)))}")
extra = vllm.get("extra_args") or []
print(f"EXTRA_ARGS={shlex.quote(' '.join(str(a) for a in extra))}")
PY
)"

echo "Starting vLLM: model=$MODEL host=$HOST port=$PORT tp=$TP"

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  $EXTRA_ARGS
