#!/usr/bin/env python3
"""Inference tests analysis page."""

import streamlit as st
import pandas as pd

from common import init_page, show_data_source_info
from components.header import render_header
from utils.visualizations import (
    plot_timeseries_auto,
    create_summary_table_infer,
)

init_page("推理测试分析 | InfiniBench", "🚀")


def main():
    render_header()
    st.markdown("## 🚀 推理性能测试分析")

    show_data_source_info()

    runs = st.session_state.data_loader.list_test_runs("infer")

    if not runs:
        st.info("未找到推理测试结果（testcase 需以 infer.* 开头）。")
        return

    # ---------- Sidebar Filters ----------
    with st.sidebar:
        st.markdown("### 🔍 筛选条件")

        frameworks = sorted(
            set((r.get("config", {}).get("framework") or "unknown") for r in runs)
        )
        selected_fw = st.multiselect("框架", frameworks, default=frameworks)

        modes = []
        for r in runs:
            tc = r.get("testcase", "")
            if "Service" in tc or "service" in tc.lower():
                modes.append("service")
            else:
                modes.append("direct")
        modes = sorted(set(modes))
        selected_modes = st.multiselect("模式", modes, default=modes)

        device_counts = sorted(set(r.get("device_used", 1) for r in runs))
        selected_dev = st.multiselect("设备数", device_counts, default=device_counts)

        only_success = st.checkbox("仅显示成功测试", value=True)

        y_log = st.checkbox("Y轴对数刻度（部分曲线更清晰）", value=False)

    def _mode_of(r):
        tc = r.get("testcase") or ""
        return "service" if ("Service" in tc or "service" in tc.lower()) else "direct"

    filtered = [
        r
        for r in runs
        if (
            not selected_fw
            or (r.get("config", {}).get("framework") or "unknown") in selected_fw
        )
        and (not selected_modes or _mode_of(r) in selected_modes)
        and (not selected_dev or r.get("device_used", 1) in selected_dev)
        and (not only_success or r.get("success"))
    ]

    st.caption(f"找到 {len(filtered)} 个推理测试")

    if not filtered:
        st.warning("没有符合条件的测试结果")
        return

    # ---------- Run Selection ----------
    options = {
        f"{r.get('config', {}).get('framework','unknown')} | {_mode_of(r)} | {r.get('device_used','?')} GPU | {r.get('time','')} | {r.get('run_id','')[:10]}": i
        for i, r in enumerate(filtered)
    }

    selected = st.multiselect(
        "选择要分析的测试运行（可多选对比）",
        list(options.keys()),
        default=list(options.keys())[:1],
    )
    if not selected:
        return

    selected_runs = []
    for k in selected:
        ri = filtered[options[k]]
        # Use path for file source, run_id for MongoDB
        identifier = ri.get("path") or ri.get("run_id")
        data = st.session_state.data_loader.load_test_result(identifier)
        ri = dict(ri)
        ri["data"] = data
        selected_runs.append(ri)

    tab1, tab2, tab3 = st.tabs(["📈 性能图表", "📊 数据表格", "🔍 详细配置"])

    # ---------- Charts ----------
    with tab1:
        st.markdown("### 指标曲线（Latency / TTFT / Throughput）")

        # layout: 3 columns
        c1, c2, c3 = st.columns(3)

        def _plot_metric(metric_name_contains: str, container):
            with container:
                if len(selected_runs) == 1:
                    run = selected_runs[0]
                    metrics = run["data"].get("metrics", [])
                    hit = next(
                        (
                            m
                            for m in metrics
                            if metric_name_contains in (m.get("name", ""))
                            and m.get("data") is not None
                        ),
                        None,
                    )
                    if hit:
                        df = hit["data"]
                        fig = plot_timeseries_auto(
                            df,
                            title=f"{hit['name']} - {run.get('config',{}).get('framework','')}",
                            y_log_scale=y_log,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"未找到 {metric_name_contains} 对应的 CSV")
                else:
                    # multi-run comparison: overlay lines
                    st.markdown(f"**对比：{metric_name_contains}**")
                    lines = []
                    for run in selected_runs:
                        hit = next(
                            (
                                m
                                for m in run["data"].get("metrics", [])
                                if metric_name_contains in (m.get("name", ""))
                                and m.get("data") is not None
                            ),
                            None,
                        )
                        if not hit:
                            continue
                        lines.append((run, hit))
                    if not lines:
                        st.info("选中的运行中没有可用数据")
                        return

                    import plotly.graph_objects as go

                    fig = go.Figure()
                    for run, hit in lines:
                        df = hit["data"]
                        xcol = df.columns[0]
                        ycol = df.columns[1] if len(df.columns) > 1 else None
                        if ycol is None:
                            continue
                        label = f"{run.get('config',{}).get('framework','unknown')}|{_mode_of(run)}|{run.get('device_used','?')}GPU"
                        fig.add_trace(
                            go.Scatter(
                                x=df[xcol], y=df[ycol], mode="lines+markers", name=label
                            )
                        )
                    fig.update_layout(
                        title=f"{metric_name_contains} 对比",
                        xaxis_title="step",
                        yaxis_title=metric_name_contains,
                        template="plotly_white",
                        height=420,
                    )
                    if y_log:
                        fig.update_yaxes(type="log")
                    st.plotly_chart(fig, use_container_width=True)

        _plot_metric("infer.compute_latency", c1)
        _plot_metric("infer.ttft", c2)
        _plot_metric("infer.direct_throughput_tps", c3)

    # ---------- Tables ----------
    with tab2:
        for run in selected_runs:
            with st.expander(f"{run.get('run_id')} - 原始数据"):
                for m in run["data"].get("metrics", []):
                    if m.get("data") is None:
                        continue
                    st.markdown(f"**{m.get('name')}**")
                    st.dataframe(m["data"], use_container_width=True, hide_index=True)

    # ---------- Config ----------
    with tab3:
        for run in selected_runs:
            with st.expander(f"{run.get('run_id')} - 配置与环境"):
                summary = create_summary_table_infer(run["data"])
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.markdown("**config**")
                st.json(run["data"].get("config", {}))
                st.markdown("**resolved**")
                st.json(run["data"].get("resolved", {}))
                st.markdown("**environment**")
                st.json(run["data"].get("environment", {}))


if __name__ == "__main__":
    main()
