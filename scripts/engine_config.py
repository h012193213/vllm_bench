#!/usr/bin/env python3
"""Shared inference-engine matrix validation and active-engine marker."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ENGINE_PATH = ROOT / "configs" / ".active_engine"

ENGINE_SECTIONS: dict[str, str] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "tensorrt_llm": "tensorrt_llm",
}

ENGINE_PACKAGES: dict[str, str] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "tensorrt_llm": "tensorrt_llm",
}


def load_matrix(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def require_engine_config(matrix: dict[str, Any], engine: str, matrix_path: Path) -> dict[str, Any]:
    if engine not in ENGINE_SECTIONS:
        raise SystemExit(f"Unknown engine '{engine}'. Expected one of: {', '.join(ENGINE_SECTIONS)}")
    key = ENGINE_SECTIONS[engine]
    section = matrix.get(key)
    if not isinstance(section, dict):
        raise SystemExit(
            f"Matrix missing required section '{key}' for engine '{engine}'. "
            f"Add a '{key}:' block to {matrix_path}."
        )
    return section


def normalize_engine_cfg(engine_cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply defaults to the self-contained per-engine matrix block."""
    cfg = dict(engine_cfg)
    cfg.setdefault("host", "0.0.0.0")
    cfg.setdefault("port", 8000)
    cfg.setdefault("tensor_parallel_size", 1)
    cfg.setdefault("max_model_len", 8192)
    cfg.setdefault("extra_args", [])
    return cfg


def write_active_engine(
    engine: str,
    matrix_path: Path,
    port: int,
    path: Path = ACTIVE_ENGINE_PATH,
) -> dict[str, Any]:
    payload = {
        "engine": engine,
        "matrix": str(matrix_path.relative_to(ROOT)) if matrix_path.is_relative_to(ROOT) else str(matrix_path),
        "port": port,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_active_engine(path: Path = ACTIVE_ENGINE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            "No active engine. Start a serve script first "
            "(e.g. configs/vllm_serve.sh or configs/sglang_serve.sh)."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid active engine marker at {path}: {exc}") from exc
    engine = payload.get("engine")
    if engine not in ENGINE_SECTIONS:
        raise SystemExit(f"Active engine marker at {path} has invalid engine: {engine!r}")
    return payload


def gpu_slug(name: str) -> str:
    lowered = name.lower()
    for prefix in ("nvidia ", "geforce "):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
    compact = re.sub(r"\s+", "", lowered)
    for pattern in (
        r"rtx\d+",
        r"h100",
        r"a100",
        r"l40s?",
        r"l4",
        r"t4",
        r"v100",
        r"mi300",
    ):
        match = re.search(pattern, compact)
        if match:
            return match.group(0)
    slug = re.sub(r"[^a-z0-9]+", "", compact)
    return slug[:32] if slug else "gpu"


def gpu_slug_from_system() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from collect_hardware import collect_gpus  # noqa: PLC0415

    gpus = collect_gpus()
    if not gpus:
        return "cpu"
    return gpu_slug(gpus[0]["name"])


def generate_run_id(engine: str) -> str:
    ts = datetime.now().strftime("%m%d-%H%M")
    return f"{gpu_slug_from_system()}-{engine}-{ts}"


def infer_engine_from_run_id(run_id: str) -> str | None:
    for engine in ("tensorrt_llm", "sglang", "vllm"):
        if f"-{engine}-" in run_id:
            return engine
    return None


def target_port(target: str) -> int | None:
    parsed = urlparse(target)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme in ("http", "https") and not parsed.port:
        return 443 if parsed.scheme == "https" else 80
    return None


def prepare_serve(engine: str, matrix_path: Path) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    engine_cfg = normalize_engine_cfg(require_engine_config(matrix, engine, matrix_path))
    port = int(engine_cfg["port"])
    write_active_engine(engine, matrix_path, port)
    return {
        "model": matrix["model"],
        "engine_cfg": engine_cfg,
    }


def emit_vllm_vars(ctx: dict[str, Any]) -> None:
    cfg = ctx["engine_cfg"]
    extra = cfg.get("extra_args") or []
    print(f"MODEL={shlex.quote(ctx['model'])}")
    print(f"HOST={shlex.quote(str(cfg['host']))}")
    print(f"PORT={shlex.quote(str(cfg['port']))}")
    print(f"TP={shlex.quote(str(cfg['tensor_parallel_size']))}")
    print(f"GPU_MEM={shlex.quote(str(cfg.get('gpu_memory_utilization', 0.90)))}")
    print(f"MAX_LEN={shlex.quote(str(cfg['max_model_len']))}")
    print(f"EXTRA_ARGS={shlex.quote(' '.join(str(a) for a in extra))}")


def emit_sglang_vars(ctx: dict[str, Any]) -> None:
    cfg = ctx["engine_cfg"]
    extra = cfg.get("extra_args") or []
    print(f"MODEL={shlex.quote(ctx['model'])}")
    print(f"HOST={shlex.quote(str(cfg['host']))}")
    print(f"PORT={shlex.quote(str(cfg['port']))}")
    print(f"TP={shlex.quote(str(cfg['tensor_parallel_size']))}")
    print(f"MAX_LEN={shlex.quote(str(cfg['max_model_len']))}")
    print(f"MEM_FRAC={shlex.quote(str(cfg.get('mem_fraction_static', 0.90)))}")
    print(f"EXTRA_ARGS={shlex.quote(' '.join(str(a) for a in extra))}")


def emit_trtllm_vars(ctx: dict[str, Any]) -> None:
    cfg = ctx["engine_cfg"]
    extra = cfg.get("extra_args") or []
    print(f"MODEL={shlex.quote(ctx['model'])}")
    print(f"HOST={shlex.quote(str(cfg['host']))}")
    print(f"PORT={shlex.quote(str(cfg['port']))}")
    print(f"TP={shlex.quote(str(cfg['tensor_parallel_size']))}")
    print(f"MAX_LEN={shlex.quote(str(cfg['max_model_len']))}")
    print(f"EXTRA_ARGS={shlex.quote(' '.join(str(a) for a in extra))}")


EMITTERS = {
    "vllm": emit_vllm_vars,
    "sglang": emit_sglang_vars,
    "tensorrt_llm": emit_trtllm_vars,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference engine matrix helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve-vars", help="Validate matrix, stamp active engine, emit shell vars")
    serve.add_argument("engine", choices=list(ENGINE_SECTIONS))
    serve.add_argument("matrix", type=Path)

    args = parser.parse_args()
    if args.command == "serve-vars":
        ctx = prepare_serve(args.engine, args.matrix.resolve())
        EMITTERS[args.engine](ctx)


if __name__ == "__main__":
    main()
