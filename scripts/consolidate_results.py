#!/usr/bin/env python3
"""Consolidate GuideLLM benchmark results into DuckDB and master CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _metric_value(dist: dict[str, Any] | None, field: str = "mean") -> float | None:
    if not dist:
        return None
    bucket = dist.get("successful") or dist.get("total")
    if not bucket:
        return None
    value = bucket.get(field)
    return float(value) if value is not None else None


def _percentile(dist: dict[str, Any] | None, key: str) -> float | None:
    if not dist:
        return None
    bucket = dist.get("successful") or dist.get("total")
    if not bucket:
        return None
    percentiles = bucket.get("percentiles") or {}
    value = percentiles.get(key)
    return float(value) if value is not None else None


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def parse_benchmark_row(
    run_id: str,
    scenario: str,
    manifest: dict[str, Any],
    bench_json: Path,
) -> list[dict[str, Any]]:
    data = json.loads(bench_json.read_text())
    rows: list[dict[str, Any]] = []

    cloud = manifest.get("cloud") or {}
    cpu = manifest.get("cpu") or {}
    gpus = manifest.get("gpu") or []
    gpu_name = gpus[0]["name"] if gpus else "none"
    gpu_vram = gpus[0].get("vram_gb") if gpus else None

    slo = {}
    matrix_path = ROOT / "configs" / "benchmark_matrix.yaml"
    if matrix_path.exists():
        import yaml

        matrix = yaml.safe_load(matrix_path.read_text())
        slo = matrix.get("slo") or {}

    for benchmark in data.get("benchmarks") or []:
        metrics = benchmark.get("metrics") or {}
        totals = metrics.get("request_totals") or {}
        successful = int(totals.get("successful") or 0)
        incomplete = int(totals.get("incomplete") or 0)
        errored = int(totals.get("errored") or 0)
        total = int(totals.get("total") or successful + incomplete + errored)

        latency = metrics.get("request_latency")
        ttft = metrics.get("time_to_first_token_ms")
        tpot = metrics.get("time_per_output_token_ms")
        itl = metrics.get("inter_token_latency_ms")
        out_tps = metrics.get("output_tokens_per_second")
        req_tps = metrics.get("requests_per_second")
        prompt_tps = metrics.get("prompt_tokens_per_second")
        total_tps = metrics.get("tokens_per_second")
        concurrency = metrics.get("request_concurrency")
        out_tok_per_iter = metrics.get("output_tokens_per_iteration")

        latency_p95 = _percentile(latency, "p95")
        ttft_p95 = _percentile(ttft, "p95")
        latency_p50 = _percentile(latency, "p50")
        ttft_p50 = _percentile(ttft, "p50")

        error_rate = (errored / total) if total else 0.0
        incomplete_rate = (incomplete / total) if total else 0.0

        slo_latency = float(slo.get("latency_p95_sec") or 2.0)
        slo_ttft = float(slo.get("ttft_p95_ms") or 500.0)
        slo_pass = (
            (latency_p95 is None or latency_p95 <= slo_latency)
            and (ttft_p95 is None or ttft_p95 <= slo_ttft)
            and error_rate <= 0.01
        )

        strategy = benchmark.get("config", {}).get("strategy", {}).get("type_")
        validation_path = bench_json.parent / "validation.json"
        validation_passed = None
        if validation_path.exists():
            validation_passed = json.loads(validation_path.read_text()).get("passed")

        rows.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "strategy": strategy,
                "cloud_provider": cloud.get("provider"),
                "cloud_region": cloud.get("region"),
                "instance_type": cloud.get("instance_type"),
                "cpu_model": cpu.get("model"),
                "cpu_cores": cpu.get("cores"),
                "cpu_ram_gb": cpu.get("ram_gb"),
                "gpu_name": gpu_name,
                "gpu_vram_gb": gpu_vram,
                "gpu_count": manifest.get("gpu_count", len(gpus)),
                "model": data.get("args", {}).get("backend_kwargs", {}).get("model")
                or data.get("args", {}).get("model"),
                "successful_requests": successful,
                "incomplete_requests": incomplete,
                "errored_requests": errored,
                "total_requests": total,
                "error_rate": error_rate,
                "incomplete_rate": incomplete_rate,
                "latency_mean_sec": _metric_value(latency, "mean"),
                "latency_median_sec": _metric_value(latency, "median"),
                "latency_p50_sec": latency_p50,
                "latency_p95_sec": latency_p95,
                "latency_p99_sec": _percentile(latency, "p99"),
                "ttft_mean_ms": _metric_value(ttft, "mean"),
                "ttft_p50_ms": ttft_p50,
                "ttft_p95_ms": ttft_p95,
                "ttft_p99_ms": _percentile(ttft, "p99"),
                "tpot_mean_ms": _metric_value(tpot, "mean"),
                "tpot_median_ms": _metric_value(tpot, "median"),
                "tpot_p50_ms": _percentile(tpot, "p50"),
                "tpot_p95_ms": _percentile(tpot, "p95"),
                "itl_mean_ms": _metric_value(itl, "mean"),
                "itl_p95_ms": _percentile(itl, "p95"),
                "output_tokens_per_sec": _metric_value(out_tps, "mean"),
                "requests_per_sec": _metric_value(req_tps, "mean"),
                "prompt_tokens_per_sec": _metric_value(prompt_tps, "mean"),
                "total_tokens_per_sec": _metric_value(total_tps, "mean"),
                "concurrency_mean": _metric_value(concurrency, "mean"),
                "output_tokens_per_iter": _metric_value(out_tok_per_iter, "mean"),
                "duration_sec": benchmark.get("duration"),
                "slo_pass": slo_pass,
                "validation_passed": validation_passed,
                "benchmark_json": str(bench_json),
                "benchmark_html": str(bench_json.parent / "benchmarks.html"),
                "hardware_manifest": str(bench_json.parent / "hardware_manifest.json"),
            }
        )
    return rows


def discover_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_dir.exists():
        return rows

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        manifest = load_manifest(run_dir / "hardware_manifest.json")

        for scenario_dir in sorted(run_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            bench_json = scenario_dir / "benchmarks.json"
            if not bench_json.exists():
                continue
            scenario_manifest = load_manifest(scenario_dir / "hardware_manifest.json") or manifest
            rows.extend(parse_benchmark_row(run_id, scenario_dir.name, scenario_manifest, bench_json))

    return rows


def consolidate(results_dir: Path, db_path: Path, csv_path: Path) -> pd.DataFrame:
    rows = discover_results(results_dir)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No benchmark results found under {results_dir}")
        df.to_csv(csv_path, index=False)
        return df

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE OR REPLACE TABLE benchmark_results AS SELECT * FROM df")
    con.close()

    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path} and {db_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate GuideLLM benchmark outputs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--db-path", type=Path, default=ROOT / "results" / "results.duckdb")
    parser.add_argument("--csv-path", type=Path, default=ROOT / "results" / "master_results.csv")
    args = parser.parse_args()

    consolidate(args.results_dir, args.db_path, args.csv_path)


if __name__ == "__main__":
    main()
