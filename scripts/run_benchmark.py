#!/usr/bin/env python3
"""Run GuideLLM benchmark scenarios from the fixed matrix config."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_config import (  # noqa: E402
    generate_run_id,
    read_active_engine,
    require_engine_config,
    target_port,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "benchmark_matrix.yaml"
DEFAULT_UPLOAD_CONFIG = ROOT / "configs" / "upload.yaml"
DEFAULT_GUIDELLM = Path("/root/bench_venv/bin/guidellm")


def load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_matrix_path() -> Path:
    raw = os.environ.get("MATRIX")
    if not raw:
        return DEFAULT_MATRIX
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def env_default_target() -> str | None:
    return (
        os.environ.get("BENCHMARK_TARGET")
        or os.environ.get("VLLM_TARGET")
        or os.environ.get("GUIDELLM_TARGET")
    )


def env_default_guidellm() -> str | None:
    return os.environ.get("GUIDELLM")


def load_matrix(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def upload_enabled(config_path: Path = DEFAULT_UPLOAD_CONFIG) -> bool:
    if not config_path.exists():
        return False
    config = yaml.safe_load(config_path.read_text()) or {}
    return bool((config.get("upload") or {}).get("enabled", False))


def maybe_upload_results(skip_upload: bool) -> None:
    if skip_upload or not upload_enabled():
        return
    script = ROOT / "scripts" / "upload_results.sh"
    if not script.exists():
        print("Upload enabled but missing scripts/upload_results.sh", file=sys.stderr)
        raise SystemExit(1)
    subprocess.run([str(script)], check=True)


def resolve_guidellm(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if DEFAULT_GUIDELLM.exists():
        return DEFAULT_GUIDELLM
    found = shutil.which("guidellm")
    if not found:
        raise SystemExit("guidellm not found; install GuideLLM or pass --guidellm")
    return Path(found)


def wait_for_health(target: str, timeout_sec: float = 300.0) -> None:
    base = target.rstrip("/")
    health_urls = [f"{base}/health", f"{base}/v1/models"]
    deadline = time.time() + timeout_sec
    last_error = "unknown"

    while time.time() < deadline:
        for url in health_urls:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if 200 <= response.status < 300:
                        print(f"Server ready: {url}")
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
        time.sleep(2)

    raise SystemExit(f"Server not ready at {target} after {timeout_sec}s: {last_error}")


def validate_run(
    benchmark_json: Path,
    gates: dict[str, Any],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    data = json.loads(benchmark_json.read_text())
    benchmarks = data.get("benchmarks") or []
    if not benchmarks:
        return False, ["no benchmarks in report"]

    for benchmark in benchmarks:
        totals = benchmark.get("metrics", {}).get("request_totals", {})
        successful = int(totals.get("successful") or 0)
        incomplete = int(totals.get("incomplete") or 0)
        errored = int(totals.get("errored") or 0)
        total = int(totals.get("total") or successful + incomplete + errored)

        label = benchmark.get("config", {}).get("strategy", {}).get("type_") or "benchmark"

        min_success = int(gates.get("min_successful_requests") or 0)
        if successful < min_success:
            issues.append(f"{label}: only {successful} successful requests (min {min_success})")

        if total > 0:
            error_rate = errored / total
            incomplete_rate = incomplete / total
            max_error = float(gates.get("max_error_rate") or 1.0)
            max_incomplete = float(gates.get("max_incomplete_rate") or 1.0)
            if error_rate > max_error:
                issues.append(f"{label}: error rate {error_rate:.2%} exceeds {max_error:.2%}")
            if incomplete_rate > max_incomplete:
                issues.append(
                    f"{label}: incomplete rate {incomplete_rate:.2%} exceeds {max_incomplete:.2%}"
                )

    return len(issues) == 0, issues


def run_scenario(
    guidellm: Path,
    matrix: dict[str, Any],
    scenario: dict[str, Any],
    output_dir: Path,
    target: str,
    max_seconds: float,
) -> int:
    g = matrix["guidellm"]
    cmd = [
        str(guidellm),
        "benchmark",
        "run",
        "--target",
        target,
        "--model",
        matrix["model"],
        "--request-format",
        g["request_format"],
        "--profile",
        scenario["profile"],
        "--data",
        g["data"],
        "--random-seed",
        str(g["random_seed"]),
        "--max-seconds",
        str(max_seconds),
        "--output-dir",
        str(output_dir),
    ]
    if scenario.get("rate") is not None:
        cmd.extend(["--rate", str(scenario["rate"])])
    rampup = scenario.get("rampup", g.get("rampup"))
    if rampup is not None and float(rampup) > 0:
        cmd.extend(["--rampup", str(rampup)])
    for output in g.get("outputs") or ["json", "csv", "html"]:
        cmd.extend(["--outputs", output])

    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser(description="Run fixed-criteria GuideLLM benchmark matrix")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run identifier (default: {gpu}-{engine}-{MMDD-HHMM} from active engine marker)",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Matrix YAML (default: MATRIX from .env or configs/benchmark_matrix.yaml)",
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--target",
        default=None,
        help="Override GuideLLM target URL (default: VLLM_TARGET from .env or matrix)",
    )
    parser.add_argument(
        "--guidellm",
        default=None,
        help="Path to guidellm binary (default: GUIDELLM from .env or PATH)",
    )
    parser.add_argument("--pilot", action="store_true", help="Use pilot settings (shorter runs)")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None, help="Hardware manifest to copy")
    parser.add_argument("--no-upload", action="store_true", help="Skip SSH upload after benchmark")
    args = parser.parse_args()

    matrix_path = args.matrix or resolve_matrix_path()
    active = read_active_engine()
    engine = active["engine"]
    guidellm = resolve_guidellm(args.guidellm or env_default_guidellm())
    matrix = load_matrix(matrix_path)
    require_engine_config(matrix, engine, matrix_path)
    g = matrix["guidellm"]
    target = args.target or env_default_target() or g["target"]

    run_id = args.run_id or generate_run_id(engine)
    print(f"Run ID: {run_id} (engine: {engine})")

    marker_port = active.get("port")
    resolved_port = target_port(target)
    if marker_port is not None and resolved_port is not None and int(marker_port) != resolved_port:
        print(
            f"Warning: active engine port {marker_port} differs from benchmark target port {resolved_port}",
            file=sys.stderr,
        )

    pilot = matrix.get("pilot") or {}
    if args.pilot:
        max_seconds = float(pilot.get("max_seconds") or 60)
        idle_seconds = float(pilot.get("scenario_idle_seconds") or 5)
        scenario_names = pilot.get("scenarios")
        gates = {**matrix.get("quality_gates", {}), **{k: v for k, v in pilot.items() if k.startswith("min_") or k.startswith("max_")}}
    else:
        max_seconds = float(g["max_seconds"])
        idle_seconds = float(g.get("scenario_idle_seconds") or 0)
        scenario_names = None
        gates = matrix.get("quality_gates") or {}

    scenarios = matrix["guidellm"]["scenarios"]
    if scenario_names:
        scenarios = [s for s in scenarios if s["name"] in scenario_names]

    run_dir = args.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or run_dir / "hardware_manifest.json"
    if not manifest_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "collect_hardware.py"),
                "--out",
                str(manifest_path),
                "--run-id",
                run_id,
                "--inference-engine",
                engine,
            ],
            check=True,
        )

    if not args.skip_health_check:
        wait_for_health(target)

    failures: list[str] = []
    for index, scenario in enumerate(scenarios):
        name = scenario["name"]
        out = run_dir / name
        out.mkdir(parents=True, exist_ok=True)

        rc = run_scenario(guidellm, matrix, scenario, out, target, max_seconds)
        if rc != 0:
            msg = f"{name}: guidellm exited with code {rc}"
            failures.append(msg)
            print(msg, file=sys.stderr)
            if not args.continue_on_failure:
                raise SystemExit(rc)

        shutil.copy2(manifest_path, out / "hardware_manifest.json")

        bench_json = out / "benchmarks.json"
        if bench_json.exists():
            ok, issues = validate_run(bench_json, gates)
            status_path = out / "validation.json"
            status_path.write_text(json.dumps({"passed": ok, "issues": issues}, indent=2) + "\n")
            if ok:
                print(f"{name}: validation passed")
            else:
                print(f"{name}: validation issues:", "; ".join(issues), file=sys.stderr)
                failures.extend(f"{name}: {issue}" for issue in issues)
        else:
            failures.append(f"{name}: missing benchmarks.json")

        if index + 1 < len(scenarios) and idle_seconds > 0:
            print(f"Idle {idle_seconds}s before next scenario...")
            time.sleep(idle_seconds)

    summary = {
        "run_id": run_id,
        "inference_engine": engine,
        "target": target,
        "pilot": args.pilot,
        "failures": failures,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    maybe_upload_results(args.no_upload)

    if failures and not args.continue_on_failure:
        raise SystemExit(f"Benchmark run completed with {len(failures)} issue(s)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
