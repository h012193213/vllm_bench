#!/usr/bin/env python3
"""Streamlit dashboard for cross-hardware GuideLLM benchmark comparison."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "results" / "results.duckdb"
DEFAULT_CSV = ROOT / "results" / "master_results.csv"


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
        st.subheader("Leaderboard (lower latency is better)")
        rank_metric = st.selectbox(
            "Rank by",
            ["latency_p95_sec", "latency_mean_sec", "ttft_p95_ms", "output_tokens_per_sec"],
            index=0,
        )
        ascending = rank_metric != "output_tokens_per_sec"
        leaderboard = filtered.sort_values(rank_metric, ascending=ascending, na_position="last")
        display_cols = [
            "run_id",
            "scenario",
            "cloud_provider",
            "instance_type",
            "cpu_model",
            "gpu_name",
            "latency_p95_sec",
            "ttft_p95_ms",
            "output_tokens_per_sec",
            "requests_per_sec",
            "error_rate",
            "slo_pass",
        ]
        st.dataframe(leaderboard[display_cols], use_container_width=True)

        chart_df = leaderboard.dropna(subset=[rank_metric])
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="run_id",
                y=rank_metric,
                color="scenario",
                barmode="group",
                title=f"{rank_metric} by run",
            )
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
        st.dataframe(run_rows, use_container_width=True)

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
        st.download_button(
            "Download filtered CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_benchmark_results.csv",
            mime="text/csv",
        )
        st.dataframe(filtered, use_container_width=True)


if __name__ == "__main__":
    main()
