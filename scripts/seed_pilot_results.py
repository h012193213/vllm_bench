#!/usr/bin/env python3
"""Seed pilot benchmark results for dashboard/consolidation verification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks.json"


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def copy_benchmark(scenario_dir: Path) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    if SOURCE.exists():
        shutil.copy2(SOURCE, scenario_dir / "benchmarks.json")
    csv = ROOT / "benchmarks.csv"
    if csv.exists():
        shutil.copy2(csv, scenario_dir / "benchmarks.csv")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source benchmark: {SOURCE}")

    runs = [
        {
            "run_id": "pilot-do-nyc1-cpu",
            "scenario": "load_rps_4",
            "manifest": {
                "run_id": "pilot-do-nyc1-cpu",
                "cloud": {
                    "provider": "digitalocean",
                    "region": "nyc1",
                    "instance_type": "s-2vcpu-4gb",
                    "instance_id": "570495571",
                },
                "cpu": {"model": "DO-Regular", "cores": 2, "architecture": "x86_64"},
                "gpu": [],
                "gpu_count": 0,
                "guidellm_version": "0.6.0",
                "notes": "Pilot seed from archived GuideLLM run; remote vLLM client",
            },
        },
        {
            "run_id": "pilot-aws-g5-a10g",
            "scenario": "load_rps_4",
            "manifest": {
                "run_id": "pilot-aws-g5-a10g",
                "cloud": {
                    "provider": "aws",
                    "region": "us-east-1",
                    "instance_type": "g5.xlarge",
                    "instance_id": "i-0seedpilot001",
                },
                "cpu": {"model": "Intel Xeon Platinum 8259CL", "cores": 4, "architecture": "x86_64"},
                "gpu": [{"name": "NVIDIA A10G", "count": 1, "vram_gb": 24.0, "driver": "535.54.03"}],
                "gpu_count": 1,
                "guidellm_version": "0.6.0",
                "notes": "Pilot seed from archived GuideLLM run; simulated second hardware target",
            },
        },
    ]

    for entry in runs:
        run_dir = ROOT / "results" / entry["run_id"]
        scenario_dir = run_dir / entry["scenario"]
        copy_benchmark(scenario_dir)
        write_manifest(run_dir / "hardware_manifest.json", entry["manifest"])
        write_manifest(scenario_dir / "hardware_manifest.json", entry["manifest"])

        validation = {"passed": False, "issues": ["pilot seed data; quality gates not re-evaluated"]}
        (scenario_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    # Add baseline_sync for DO run using same source with different scenario name.
    sync_dir = ROOT / "results" / "pilot-do-nyc1-cpu" / "baseline_sync"
    copy_benchmark(sync_dir)
    manifest = json.loads((ROOT / "results" / "pilot-do-nyc1-cpu" / "hardware_manifest.json").read_text())
    write_manifest(sync_dir / "hardware_manifest.json", manifest)

    print("Seeded pilot results under results/pilot-do-nyc1-cpu and results/pilot-aws-g5-a10g")


if __name__ == "__main__":
    main()
