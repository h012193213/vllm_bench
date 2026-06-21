#!/usr/bin/env bash
# Push all local benchmark results to the central analysis host via rsync over SSH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/configs/upload.yaml"
RESULTS_DIR="${ROOT}/results"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: upload_results.sh [--dry-run]"
      echo "Reads fixed settings from configs/upload.yaml"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing upload config: $CONFIG" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found; install rsync" >&2
  exit 1
fi

eval "$(python3 - "$CONFIG" <<'PY'
import shlex
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()).get("upload") or {}
enabled = cfg.get("enabled", False)
print(f"UPLOAD_ENABLED={1 if enabled else 0}")
print(f"CENTRAL_HOST={shlex.quote(str(cfg.get('central_host') or ''))}")
print(f"CENTRAL_USER={shlex.quote(str(cfg.get('central_user') or 'root'))}")
print(f"CENTRAL_PATH={shlex.quote(str(cfg.get('central_path') or '/root/vllm_bench/results'))}")
ssh_key = cfg.get("ssh_key")
print(f"SSH_KEY={shlex.quote(str(ssh_key)) if ssh_key else ''}")
PY
)"

if [[ "$UPLOAD_ENABLED" -eq 0 ]]; then
  echo "Upload disabled in $CONFIG"
  exit 0
fi

if [[ -z "$CENTRAL_HOST" || "$CENTRAL_HOST" == "your-analysis-vm.example.com" ]]; then
  echo "Set central_host in $CONFIG before enabling upload" >&2
  exit 1
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "No results directory: $RESULTS_DIR" >&2
  exit 1
fi

REMOTE="${CENTRAL_USER}@${CENTRAL_HOST}:${CENTRAL_PATH%/}/"
RSYNC_OPTS=(-avz --progress
  --exclude 'master_results.csv'
  --exclude 'results.duckdb'
  --exclude '__pycache__'
)

if [[ -n "$SSH_KEY" ]]; then
  if [[ ! -f "$SSH_KEY" ]]; then
    echo "SSH key not found: $SSH_KEY" >&2
    exit 1
  fi
  RSYNC_OPTS+=(-e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new")
else
  RSYNC_OPTS+=(-e "ssh -o StrictHostKeyChecking=accept-new")
fi

echo "Uploading $RESULTS_DIR -> $REMOTE"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run:"
  echo rsync "${RSYNC_OPTS[@]}" "$RESULTS_DIR/" "$REMOTE"
  exit 0
fi

rsync "${RSYNC_OPTS[@]}" "$RESULTS_DIR/" "$REMOTE"
echo "Upload complete."
