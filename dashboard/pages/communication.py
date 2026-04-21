#!/usr/bin/env python3
"""Communication tests analysis page."""

import streamlit as st
import pandas as pd

from common import init_page, show_data_source_info
from components.header import render_header
from utils.data_loader import get_friendly_size
from utils.metrics import extract_core_metrics
from utils.visualizations import (
    plot_metric_vs_size,
    plot_comparison_matrix,
    create_gauge_chart,
    plot_timeseries_auto,
    create_summary_table_infer,
    create_summary_table_comm,
)

init_page("通信测试分析 | InfiniBench", "🔗")


def main():
    """Main function for communication tests page."""
    render_header()
    st.markdown("## 🔗 通信性能测试分析")

    show_data_source_info()

    try:
        # Load communication test results
        comm_runs = st.session_state.data_loader.list_test_runs("comm")

        if not comm_runs:
            st.info("未找到通信测试结果")
            st.info("请先运行通信测试或检查测试结果目录")
            return

        # Sidebar filters
        with st.sidebar:
            st.markdown("### 🔍 筛选条件")

            # Operation filter
            operations = list(set(r["operation"] for r in comm_runs))
            selected_ops = st.multiselect(
                "选择算子",
                options=operations,
                default=operations[:1] if operations else [],
            )

            # Device count filter
            device_counts = list(set(r["device_used"] for r in comm_runs))
            if device_counts:
                selected_devices = st.multiselect(
                    "选择设备数",
                    options=sorted(device_counts),
                    default=sorted(device_counts)[:1] if device_counts else [],
                )

            # Status filter
            show_success = st.checkbox("仅显示成功测试", value=True)

            # Apply filter
            filtered_runs = [
                r
                for r in comm_runs
                if (not selected_ops or r["operation"] in selected_ops)
                and (not selected_devices or r["device_used"] in selected_devices)
                and (not show_success or r["success"])
            ]

            st.markdown(f"**找到 {len(filtered_runs)} 个测试**")

            # Visualization options
            st.markdown("---")
            st.markdown("### 📊 图表选项")
            y_log_scale = st.checkbox("Y轴使用对数刻度", value=False)

        if not filtered_runs:
            st.warning("没有符合条件的测试结果")
            return

        # Run selector
        st.markdown("### 选择测试运行")

        # Run ID Fuzzy search (really works)
        run_id_kw = st.text_input(
            "🔎 Run ID 模糊搜索（支持前缀 / 子串）",
            placeholder="例如：20240109 / abcd1234",
        )

        if run_id_kw:
            filtered_runs = [
                r for r in filtered_runs if run_id_kw in (r.get("run_id") or "")
            ]

        if not filtered_runs:
            st.warning("没有符合 Run ID 条件的测试结果")
            return

        # Create selection options
        run_options = {
            f"{r['operation']} ({r['device_used']} GPUs) - {r['time']}": i
            for i, r in enumerate(filtered_runs)
        }

        selected_indices = st.multiselect(
            "选择要分析的测试运行（可多选进行对比）",
            options=list(run_options.keys()),
            default=list(run_options.keys())[:1] if run_options else [],
            help="选择多个测试运行可以进行性能对比",
        )

        if not selected_indices:
            return

        # Load selected runs
        selected_runs = []
        for name in selected_indices:
            idx = run_options[name]
            run_info = filtered_runs[idx]
            # Use path for file source, run_id for MongoDB
            identifier = run_info.get("path") or run_info.get("run_id")
            result = st.session_state.data_loader.load_test_result(identifier)
            run_info["data"] = result

            selected_runs.append(run_info)

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 性能图表", "📊 数据表格", "🔍 详细配置"])

        with tab1:
            # Performance charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 带宽分析")
                if len(selected_runs) == 1:
                    # Single run - detailed view
                    run = selected_runs[0]
                    metrics = run.get("data", {}).get("metrics", [])
                    for metric in metrics:
                        if (
                            metric.get("name") == "comm.bandwidth"
                            and metric.get("data") is not None
                        ):
                            df = metric["data"]
                            fig = plot_metric_vs_size(
                                df=df,
                                metric_type="bandwidth",
                                title=f"带宽分析 - {run['operation']}",
                                y_log_scale=y_log_scale,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            break
                else:
                    # Multiple runs - comparison
                    fig = plot_comparison_matrix(
                        selected_runs, "bandwidth", y_log_scale
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 延迟分析")
                if len(selected_runs) == 1:
                    run = selected_runs[0]
                    metrics = run.get("data", {}).get("metrics", [])
                    for metric in metrics:
                        if (
                            metric.get("name") == "comm.latency"
                            and metric.get("data") is not None
                        ):
                            df = metric["data"]
                            fig = plot_metric_vs_size(
                                df=df,
                                metric_type="latency",
                                title=f"延迟分析 - {run['operation']}",
                                y_log_scale=y_log_scale,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            break
                else:
                    fig = plot_comparison_matrix(selected_runs, "latency", y_log_scale)
                    st.plotly_chart(fig, use_container_width=True)

            if len(selected_runs) == 1:
                st.markdown("#### 关键指标")
                run = selected_runs[0]
                core = extract_core_metrics(run)

                # First Line: numerical indicators
                cols = st.columns(3)
                cols[0].metric(
                    "峰值带宽",
                    f"{core['bandwidth_gbps']:.2f} GB/s"
                    if core["bandwidth_gbps"]
                    else "-",
                )
                cols[1].metric(
                    "平均延迟",
                    f"{core['latency_us']:.2f} μs" if core["latency_us"] else "-",
                )
                cols[2].metric(
                    "测试耗时",
                    f"{core['duration_ms']:.2f} ms" if core["duration_ms"] else "-",
                )

                cols = st.columns(3)

                with cols[0]:
                    # Find max bandwidth
                    max_bw = 0
                    for metric in run.get("data", {}).get("metrics", []):
                        if (
                            metric.get("name") == "comm.bandwidth"
                            and metric.get("data") is not None
                        ):
                            df = metric["data"]
                            if "bandwidth_gbs" in df.columns:
                                max_bw = df["bandwidth_gbs"].max()
                                fig = create_gauge_chart(
                                    max_bw,
                                    300,  # Theoretical max for A100 NVLink
                                    "峰值带宽",
                                    "blue",
                                    "GB/s",
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                break

                with cols[1]:
                    # Find average latency
                    avg_lat = 0
                    for metric in run.get("data", {}).get("metrics", []):
                        if (
                            metric.get("name") == "comm.latency"
                            and metric.get("data") is not None
                        ):
                            df = metric["data"]
                            if "latency_us" in df.columns:
                                avg_lat = df["latency_us"].mean()
                                fig = create_gauge_chart(
                                    avg_lat,
                                    1000,  # Reference: 1000 µs
                                    "平均延迟",
                                    "red",
                                    "µs",
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                break

                with cols[2]:
                    # Extract duration
                    duration = 0
                    for metric in run.get("data", {}).get("metrics", []):
                        if metric.get("name") == "comm.duration":
                            duration = metric.get("value", 0)
                            break

                    if duration > 0:
                        fig = create_gauge_chart(
                            duration,
                            duration * 2,  # Scale to show progress
                            "测试耗时",
                            "green",
                            "ms",
                        )
                        st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Data tables
            for run in selected_runs:
                with st.expander(
                    f"{run['operation']} ({run['device_used']} GPUs) - 原始数据"
                ):
                    metrics = run.get("data", {}).get("metrics", [])
                    for metric in metrics:
                        if metric.get("data") is not None:
                            df = metric["data"].copy()
                            # Add friendly size column
                            if "size_bytes" in df.columns:
                                df["size_friendly"] = df["size_bytes"].apply(
                                    get_friendly_size
                                )

                            st.markdown(f"**{metric['name']}**")
                            st.dataframe(df, use_container_width=True, hide_index=True)

        with tab3:
            # Configuration details
            for run in selected_runs:
                with st.expander(
                    f"{run['operation']} ({run['device_used']} GPUs) - 配置详情"
                ):
                    # Create summary table
                    summary_df = create_summary_table_comm(run.get("data", {}))
                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "指标": st.column_config.Column(width="medium"),
                            "数值": st.column_config.Column(width="large"),
                        },
                    )

                    # Show full config
                    st.markdown("**完整配置:**")
                    st.json(run.get("data", {}).get("config", {}))

                    # Show resolved info if available
                    if run.get("data", {}).get("resolved"):
                        st.markdown("**执行详情:**")
                        st.json(run.get("data", {}).get("resolved", {}))

    except Exception as e:
        st.error(f"加载通信测试数据失败: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
