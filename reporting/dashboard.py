#!/usr/bin/env python3
"""Streamlit dashboard for cross-hardware GuideLLM benchmark comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "results" / "results.duckdb"
DEFAULT_CSV = ROOT / "results" / "master_results.csv"

HIGHER_IS_BETTER = {
    "output_tokens_per_sec",
    "requests_per_sec",
    "prompt_tokens_per_sec",
    "total_tokens_per_sec",
    "successful_requests",
    "output_tokens_per_iter",
    "total_requests",
    "duration_sec",
    "cpu_cores",
    "cpu_ram_gb",
    "gpu_count",
    "gpu_vram_gb",
}


@dataclass(frozen=True)
class ColumnMeta:
    label: str
    help: str
    kind: str = "number"  # number | text | bool
    default_visible: bool = False


COLUMN_META: dict[str, ColumnMeta] = {
    "run_id": ColumnMeta("Run", "Unique benchmark run identifier.", "text", True),
    "scenario": ColumnMeta("Scenario", "Fixed-load test profile from the benchmark matrix.", "text", True),
    "strategy": ColumnMeta("Load strategy", "GuideLLM scheduling strategy (sync, constant RPS, concurrent).", "text", False),
    "cloud_provider": ColumnMeta("Cloud", "Cloud or hosting provider (auto-detected or from .env).", "text", True),
    "cloud_region": ColumnMeta("Region", "Cloud region or datacenter.", "text", False),
    "instance_type": ColumnMeta("Instance type", "Cloud instance SKU or size name.", "text", False),
    "cpu_model": ColumnMeta("CPU model", "Processor model reported by the host.", "text", False),
    "cpu_cores": ColumnMeta("CPU cores", "Number of logical CPUs on the host.", "number", False),
    "cpu_ram_gb": ColumnMeta("CPU RAM (GB)", "Total system memory in gigabytes.", "number", False),
    "gpu_name": ColumnMeta("GPU", "NVIDIA GPU model name.", "text", True),
    "gpu_count": ColumnMeta("GPU count", "Number of GPUs on the host.", "number", False),
    "gpu_vram_gb": ColumnMeta("VRAM (GB)", "GPU memory in gigabytes (first GPU).", "number", False),
    "model": ColumnMeta("LLM model", "Model served by vLLM during the benchmark.", "text", False),
    "latency_mean_sec": ColumnMeta("Latency mean (s)", "Mean end-to-end request time in seconds.", "number", False),
    "latency_median_sec": ColumnMeta("Latency median (s)", "Median end-to-end request time in seconds.", "number", False),
    "latency_p50_sec": ColumnMeta("Latency P50 (s)", "50th percentile request latency in seconds.", "number", False),
    "latency_p95_sec": ColumnMeta("Latency P95 (s)", "95th percentile request latency in seconds.", "number", True),
    "latency_p99_sec": ColumnMeta("Latency P99 (s)", "99th percentile request latency in seconds.", "number", True),
    "ttft_mean_ms": ColumnMeta("TTFT mean (ms)", "Mean time to first token in milliseconds.", "number", False),
    "ttft_p50_ms": ColumnMeta("TTFT P50 (ms)", "50th percentile time to first token in milliseconds.", "number", False),
    "ttft_p95_ms": ColumnMeta("TTFT P95 (ms)", "95th percentile time to first token in milliseconds.", "number", True),
    "ttft_p99_ms": ColumnMeta("TTFT P99 (ms)", "99th percentile time to first token in milliseconds.", "number", True),
    "tpot_mean_ms": ColumnMeta("TPOT mean (ms)", "Mean time per output token in milliseconds.", "number", True),
    "tpot_median_ms": ColumnMeta("TPOT median (ms)", "Median time per output token in milliseconds.", "number", False),
    "tpot_p50_ms": ColumnMeta("TPOT P50 (ms)", "50th percentile time per output token in milliseconds.", "number", False),
    "tpot_p95_ms": ColumnMeta("TPOT P95 (ms)", "95th percentile time per output token in milliseconds.", "number", False),
    "itl_mean_ms": ColumnMeta("Token gap mean (ms)", "Mean inter-token latency in milliseconds.", "number", False),
    "itl_p95_ms": ColumnMeta("Token gap P95 (ms)", "95th percentile inter-token latency in milliseconds.", "number", True),
    "output_tokens_per_sec": ColumnMeta("Output tok/s", "Mean output token throughput.", "number", True),
    "requests_per_sec": ColumnMeta("Requests/s", "Mean completed requests per second.", "number", True),
    "prompt_tokens_per_sec": ColumnMeta("Prompt tok/s", "Mean prompt token throughput.", "number", True),
    "total_tokens_per_sec": ColumnMeta("Total tok/s", "Mean combined token throughput.", "number", True),
    "concurrency_mean": ColumnMeta("Concurrency", "Mean in-flight request concurrency.", "number", True),
    "output_tokens_per_iter": ColumnMeta("Tok / stream step", "Mean output tokens per streaming iteration.", "number", False),
    "successful_requests": ColumnMeta("Completed", "Number of successfully completed requests.", "number", False),
    "incomplete_requests": ColumnMeta("Incomplete", "Requests that did not finish (often saturated GPU).", "number", False),
    "errored_requests": ColumnMeta("Errors", "Requests that returned an error.", "number", False),
    "total_requests": ColumnMeta("Total requests", "All requests attempted during the scenario.", "number", False),
    "error_rate": ColumnMeta("Error rate", "Fraction of requests that errored (0–1).", "number", True),
    "incomplete_rate": ColumnMeta("Incomplete rate", "Fraction of requests left incomplete (0–1).", "number", True),
    "duration_sec": ColumnMeta("Duration (s)", "Benchmark scenario duration in seconds.", "number", False),
    "slo_pass": ColumnMeta("SLO pass", "Whether latency, TTFT, and error rate met configured SLO targets.", "bool", True),
    "validation_passed": ColumnMeta("Quality gate", "Whether run passed configured quality gates.", "bool", False),
}

RANK_METRICS = [
    key for key, meta in COLUMN_META.items() if meta.kind == "number" and key not in {
        "cpu_cores", "cpu_ram_gb", "gpu_count", "gpu_vram_gb", "duration_sec",
    }
]

DEFAULT_VISIBLE = [key for key, meta in COLUMN_META.items() if meta.default_visible]
OPTIONAL_COLUMNS = [key for key in COLUMN_META if key not in DEFAULT_VISIBLE]


@st.cache_data
def load_data(db_path: str, csv_path: str) -> pd.DataFrame:
    db = Path(db_path)
    csv = Path(csv_path)
    if db.exists():
        con = duckdb.connect(str(db), read_only=True)
        try:
            return con.execute("SELECT * FROM benchmark_results").df()
        finally:
            con.close()
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def available_rank_metrics(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [(col, COLUMN_META[col].label) for col in RANK_METRICS if col in df.columns and col in COLUMN_META]


def is_higher_better(metric: str) -> bool:
    return metric in HIGHER_IS_BETTER


def build_column_config(columns: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for col in columns:
        meta = COLUMN_META.get(col)
        if not meta:
            continue
        if meta.kind == "number":
            config[col] = st.column_config.NumberColumn(
                meta.label,
                help=meta.help,
                format="%.2f",
            )
        elif meta.kind == "bool":
            config[col] = st.column_config.CheckboxColumn(
                meta.label,
                help=meta.help,
            )
        else:
            config[col] = st.column_config.TextColumn(
                meta.label,
                help=meta.help,
            )
    return config


def format_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        meta = COLUMN_META.get(col)
        if meta and meta.kind == "number":
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    return out


def render_table(df: pd.DataFrame, columns: list[str]) -> None:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        st.info("No columns to display.")
        return
    formatted = format_numeric_columns(df[cols], cols)
    st.dataframe(
        formatted,
        use_container_width=True,
        column_config=build_column_config(cols),
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="GuideLLM Hardware Benchmark", layout="wide")
    st.title("GuideLLM Cross-Hardware Benchmark")
    st.caption("Compare fixed-load SLO performance across GPU, CPU, and cloud platforms.")

    db_path = st.sidebar.text_input("DuckDB path", str(DEFAULT_DB))
    csv_path = st.sidebar.text_input("CSV fallback path", str(DEFAULT_CSV))

    if st.sidebar.button("Reload data"):
        load_data.clear()

    df = load_data(db_path, csv_path)
    if df.empty:
        st.warning(
            "No consolidated results found. Run:\n\n"
            "`python scripts/consolidate_results.py --results-dir results/`"
        )
        return

    scenarios = sorted(df["scenario"].dropna().unique())
    providers = sorted(df["cloud_provider"].dropna().unique())
    gpus = sorted(df["gpu_name"].dropna().unique())

    selected_scenarios = st.sidebar.multiselect("Scenarios", scenarios, default=scenarios)
    selected_providers = st.sidebar.multiselect("Cloud providers", providers, default=providers)
    selected_gpus = st.sidebar.multiselect("GPU models", gpus, default=gpus)

    optional_choices = {
        col: COLUMN_META[col].label for col in OPTIONAL_COLUMNS if col in df.columns and col in COLUMN_META
    }
    extra_cols = st.sidebar.multiselect(
        "Additional columns",
        list(optional_choices.keys()),
        default=[],
        format_func=lambda col: optional_choices[col],
        help="Optional hardware and detail columns. Hover column headers in the table for explanations.",
    )

    display_columns = DEFAULT_VISIBLE + [col for col in extra_cols if col not in DEFAULT_VISIBLE]
    display_columns = [col for col in display_columns if col in df.columns]

    filtered = df[
        df["scenario"].isin(selected_scenarios)
        & df["cloud_provider"].isin(selected_providers)
        & df["gpu_name"].isin(selected_gpus)
    ].copy()

    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    tab_leaderboard, tab_slo, tab_detail, tab_export = st.tabs(
        ["Leaderboard", "SLO Heatmap", "Drill-down", "Export"]
    )

    with tab_leaderboard:
        rank_options = available_rank_metrics(filtered)
        if not rank_options:
            st.warning("No rankable metric columns found in consolidated data.")
            return

        rank_cols = [col for col, _ in rank_options]
        rank_labels = {col: label for col, label in rank_options}
        default_metric = "latency_p95_sec" if "latency_p95_sec" in rank_cols else rank_cols[0]

        st.subheader("Leaderboard")
        rank_metric = st.selectbox(
            "Rank by",
            rank_cols,
            index=rank_cols.index(default_metric),
            format_func=lambda col: rank_labels[col],
        )
        ascending = not is_higher_better(rank_metric)
        direction = "lower is better" if ascending else "higher is better"
        st.caption(f"Sorting: {rank_labels[rank_metric]} ({direction}). Hover column headers for metric explanations.")

        leaderboard = filtered.sort_values(rank_metric, ascending=ascending, na_position="last")
        render_table(leaderboard, display_columns)

        chart_df = leaderboard.dropna(subset=[rank_metric])
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="run_id",
                y=rank_metric,
                color="scenario",
                barmode="group",
                title=rank_labels[rank_metric],
                labels={rank_metric: rank_labels[rank_metric]},
            )
            fig.update_yaxes(tickformat=".2f")
            st.plotly_chart(fig, use_container_width=True)

    with tab_slo:
        st.subheader("SLO pass/fail heatmap")
        heatmap_df = (
            filtered.groupby(["run_id", "scenario"], as_index=False)["slo_pass"]
            .max()
            .assign(slo_pass=lambda d: d["slo_pass"].map({True: "pass", False: "fail"}))
        )
        pivot = heatmap_df.pivot(index="run_id", columns="scenario", values="slo_pass")
        st.dataframe(pivot, use_container_width=True)

        numeric = filtered.copy()
        numeric["slo_pass_num"] = numeric["slo_pass"].astype(float)
        fig = px.density_heatmap(
            numeric,
            x="scenario",
            y="run_id",
            z="slo_pass_num",
            histfunc="avg",
            title="SLO pass rate (1=pass, 0=fail)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_detail:
        st.subheader("Run drill-down")
        run_ids = sorted(filtered["run_id"].unique())
        selected_run = st.selectbox("Run ID", run_ids)
        run_rows = filtered[filtered["run_id"] == selected_run]
        detail_cols = [col for col in df.columns if col in COLUMN_META and not col.endswith("_json")]
        detail_cols = [col for col in detail_cols if col not in {"benchmark_json", "benchmark_html", "hardware_manifest"}]
        render_table(run_rows, detail_cols)

        for _, row in run_rows.iterrows():
            html_path = Path(str(row.get("benchmark_html") or ""))
            if html_path.exists():
                st.markdown(f"**{row['scenario']}** — [Open GuideLLM HTML report]({html_path.as_uri()})")
            manifest_path = Path(str(row.get("hardware_manifest") or ""))
            if manifest_path.exists():
                with st.expander(f"Hardware manifest: {row['scenario']}"):
                    st.json(json.loads(manifest_path.read_text()))

    with tab_export:
        st.subheader("Master export")
        export_df = format_numeric_columns(filtered, [col for col in filtered.columns if col in COLUMN_META])
        renamed = export_df.rename(columns={col: COLUMN_META[col].label for col in export_df.columns if col in COLUMN_META})
        st.download_button(
            "Download filtered CSV",
            renamed.to_csv(index=False).encode("utf-8"),
            file_name="filtered_benchmark_results.csv",
            mime="text/csv",
        )
        render_table(filtered, display_columns)


if __name__ == "__main__":
    main()
