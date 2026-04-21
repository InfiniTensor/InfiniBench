#!/usr/bin/env python3
"""Training tests analysis page."""

import streamlit as st

from common import init_page
from components.header import render_header
from utils.training_utils import (
    load_training_runs,
    filter_runs,
    create_run_options,
    load_selected_runs,
    create_training_summary,
)
from utils.visualizations import (
    render_performance_curves,
    render_throughput_comparison,
    render_data_tables,
    render_config_details,
)

init_page("训练测试分析 | InfiniBench", "🏋️")


def main():
    render_header()
    st.markdown("## 🏋️ 训练性能测试分析")

    dl = st.session_state.data_loader
    runs = load_training_runs(dl)

    if not runs:
        st.info("未找到训练测试结果\n请将训练测试结果放在 output/train/ 或 output/training/ 目录下")
        return

    # Sidebar Filters
    with st.sidebar:
        st.markdown("### 🔍 筛选条件")

        frameworks = sorted(
            {r.get("config", {}).get("framework", "unknown") for r in runs}
        )
        models = sorted({r.get("config", {}).get("model", "unknown") for r in runs})
        device_counts = sorted({r.get("device_used", 1) for r in runs})

        selected_fw = st.multiselect("框架", frameworks, default=frameworks)
        selected_models = st.multiselect("模型", models, default=models)
        selected_dev = st.multiselect("设备数", device_counts, default=device_counts)
        only_success = st.checkbox("仅显示成功测试", value=True)

        st.markdown("---")
        st.markdown("### 📈 图表选项")
        y_log = st.checkbox("Y轴对数刻度", value=False)
        smoothing = st.slider("平滑窗口", 1, 50, 5, help="对曲线进行移动平均平滑")

    # Apply filters
    filtered = filter_runs(
        runs, selected_fw, selected_models, selected_dev, only_success
    )
    st.caption(f"找到 {len(filtered)} 个训练测试")

    if not filtered:
        st.warning("没有符合条件的测试结果")
        return

    # Run Selection
    options = create_run_options(filtered)
    selected = st.multiselect(
        "选择要分析的测试运行（可多选对比）",
        list(options.keys()),
        default=list(options.keys())[: min(3, len(options))],
    )

    if not selected:
        return

    # Load selected runs
    selected_runs = load_selected_runs(dl, filtered, options, selected)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 性能曲线", "📊 吞吐量对比", "📋 数据表格", "🔍 详细配置"])

    with tab1:
        render_performance_curves(selected_runs, smoothing, y_log)
    with tab2:
        render_throughput_comparison(selected_runs)
    with tab3:
        render_data_tables(selected_runs)
    with tab4:
        render_config_details(selected_runs, create_training_summary)


if __name__ == "__main__":
    main()
