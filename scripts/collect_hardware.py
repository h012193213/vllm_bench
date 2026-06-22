#!/usr/bin/env python3
"""Collect GPU, CPU, and cloud metadata for benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _fetch(url: str, headers: dict[str, str] | None = None, timeout: float = 2.0) -> str | None:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace").strip()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _package_version(module: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(module)
    except Exception:
        pass

    bench_venv = Path("/root/bench_venv")
    if module == "guidellm":
        candidate = bench_venv / "bin" / "guidellm"
        if candidate.exists():
            output = _run([str(candidate), "--version"])
            if output:
                return output.split()[-1]
    return None


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


def apply_cloud_env_fallback(cloud: dict[str, Any]) -> dict[str, Any]:
    """Use CLOUD_* values from .env when metadata auto-detect is insufficient."""
    provider = os.environ.get("CLOUD_PROVIDER")
    if not provider:
        return cloud

    cloud["provider"] = provider
    for field, env_key in (
        ("region", "CLOUD_REGION"),
        ("instance_type", "CLOUD_INSTANCE_TYPE"),
        ("instance_id", "CLOUD_INSTANCE_ID"),
        ("availability_zone", "CLOUD_AVAILABILITY_ZONE"),
    ):
        value = os.environ.get(env_key)
        if value:
            cloud[field] = value
    cloud["source"] = "env"
    return cloud


def collect_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,count",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return gpus

    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name, memory, driver = parts[0], parts[1], parts[2]
        count = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 1
        try:
            vram_gb = round(float(memory) / 1024, 2)
        except ValueError:
            vram_gb = None
        gpus.append(
            {
                "name": name,
                "count": count,
                "vram_gb": vram_gb,
                "driver": driver,
            }
        )
    return gpus


def collect_cpu() -> dict[str, Any]:
    cpu: dict[str, Any] = {"cores": None, "model": None, "ram_gb": None, "architecture": platform.machine()}
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        for line in meminfo_path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                try:
                    kb = int(line.split()[1])
                    cpu["ram_gb"] = round(kb / 1024 / 1024, 2)
                except (IndexError, ValueError):
                    pass
                break

    lscpu = _run(["lscpu"])
    if not lscpu:
        return cpu

    for line in lscpu.splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key == "Model name":
            cpu["model"] = value
        elif key == "CPU(s)":
            try:
                cpu["cores"] = int(value)
            except ValueError:
                pass
    return cpu


def collect_cloud() -> dict[str, Any]:
    cloud: dict[str, Any] = {
        "provider": "unknown",
        "region": None,
        "instance_type": None,
        "instance_id": None,
        "availability_zone": None,
    }

    aws_token = _fetch("http://169.254.169.254/latest/api/token", headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
    aws_headers = {"X-aws-ec2-metadata-token": aws_token} if aws_token else None

    instance_type = _fetch("http://169.254.169.254/latest/meta-data/instance-type", headers=aws_headers)
    if instance_type and instance_type != "not found":
        cloud["provider"] = "aws"
        cloud["instance_type"] = instance_type
        cloud["instance_id"] = _fetch("http://169.254.169.254/latest/meta-data/instance-id", headers=aws_headers)
        cloud["region"] = _fetch("http://169.254.169.254/latest/meta-data/placement/region", headers=aws_headers)
        cloud["availability_zone"] = _fetch(
            "http://169.254.169.254/latest/meta-data/placement/availability-zone", headers=aws_headers
        )
        cloud["source"] = "metadata"
        return cloud

    gcp_headers = {"Metadata-Flavor": "Google"}
    gcp_instance = _fetch("http://metadata.google.internal/computeMetadata/v1/instance/name", headers=gcp_headers)
    if gcp_instance:
        cloud["provider"] = "gcp"
        cloud["instance_id"] = gcp_instance
        cloud["instance_type"] = _fetch(
            "http://metadata.google.internal/computeMetadata/v1/instance/machine-type", headers=gcp_headers
        )
        if cloud["instance_type"]:
            cloud["instance_type"] = cloud["instance_type"].rsplit("/", 1)[-1]
        cloud["region"] = _fetch(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone", headers=gcp_headers
        )
        if cloud["region"]:
            cloud["region"] = cloud["region"].rsplit("/", 1)[-1]
        cloud["source"] = "metadata"
        return cloud

    do_id = _fetch("http://169.254.169.254/metadata/v1/id")
    if do_id:
        cloud["provider"] = "digitalocean"
        cloud["instance_id"] = do_id
        cloud["region"] = _fetch("http://169.254.169.254/metadata/v1/region")
        cloud["instance_type"] = _fetch("http://169.254.169.254/metadata/v1/size")
        cloud["source"] = "metadata"
        return cloud

    hostname = platform.node()
    if hostname:
        cloud["provider"] = "local"
        cloud["instance_id"] = hostname
        cloud["source"] = "hostname"
    return cloud


def collect_manifest(run_id: str | None = None, notes: str | None = None) -> dict[str, Any]:
    load_env_file()
    cloud = apply_cloud_env_fallback(collect_cloud())

    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        provider = cloud.get("provider") or "local"
        region = cloud.get("region") or "unknown"
        run_id = f"{ts}-{provider}-{region}"

    gpus = collect_gpus()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "cloud": cloud,
        "cpu": collect_cpu(),
        "gpu": gpus,
        "gpu_count": len(gpus),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "guidellm_version": _package_version("guidellm"),
        "vllm_version": _package_version("vllm"),
        "hostname": platform.node(),
        "notes": notes or "client colocated on same VM when possible",
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hardware metadata for benchmark runs")
    parser.add_argument("--out", type=Path, required=True, help="Output path for hardware_manifest.json")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    parser.add_argument("--notes", default=None, help="Optional free-form notes")
    args = parser.parse_args()

    manifest = collect_manifest(run_id=args.run_id, notes=args.notes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
