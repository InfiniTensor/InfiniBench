#!/usr/bin/env python3
"""Main Streamlit application for InfiniBench dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional
from infinibench.common.constants import AcceleratorType

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from components.header import render_header
from utils.data_loader import InfiniBenchDataLoader
from common import show_data_source_info

# Page configuration
st.set_page_config(
    page_title="InfiniBench Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "data_loader" not in st.session_state:
    st.session_state.data_loader = InfiniBenchDataLoader()
if "selected_accelerators" not in st.session_state:
    st.session_state.selected_accelerators = []
if "use_mongodb" not in st.session_state:
    st.session_state.use_mongodb = False


def parse_timestamp(ts) -> Optional[datetime]:
    """Parse timestamp, support multiple formats"""
    try:
        ts_str = str(ts)
        if "_" in ts_str and len(ts_str) == 15:
            return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def format_time(ts) -> str:
    """Format timestamp as display string"""
    dt = parse_timestamp(ts)
    if dt:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)[:19] if ts else "未知"


def main():
    render_header()

    with st.sidebar:
        st.markdown("## ⚙️ 设置")

        use_mongodb = st.toggle(
            "使用 MongoDB",
            value=st.session_state.use_mongodb,
            help="切换到 MongoDB 数据源（需要 MongoDB 服务运行中）",
        )

        if use_mongodb != st.session_state.use_mongodb:
            st.session_state.use_mongodb = use_mongodb
            if use_mongodb:
                st.session_state.data_loader = InfiniBenchDataLoader(
                    use_mongodb=True, fallback_to_files=True
                )
            else:
                st.session_state.data_loader = InfiniBenchDataLoader()

        show_data_source_info(style="sidebar")
        st.markdown("---")

        results_dir = st.text_input(
            "测试结果目录", value="./output", help="包含 JSON/CSV 测试结果的目录"
        )

        if not use_mongodb and results_dir != str(
            st.session_state.data_loader.results_dir
        ):
            st.session_state.data_loader = InfiniBenchDataLoader(results_dir)

        auto_refresh = st.toggle("自动刷新", value=False)
        if auto_refresh:
            st.rerun()

        st.markdown("---")
        st.markdown("## 🧠 筛选条件")

        ACCELERATOR_OPTIONS = ["cpu"] + [a.value for a in AcceleratorType]
        ACCELERATOR_LABELS = {
            "cpu": "CPU",
            AcceleratorType.NVIDIA.value: "NVIDIA",
            AcceleratorType.AMD.value: "AMD",
            AcceleratorType.ASCEND.value: "昇腾 NPU",
            AcceleratorType.CAMBRICON.value: "寒武纪 MLU",
            AcceleratorType.GENERIC.value: "Generic",
        }

        selected_accs = st.multiselect(
            "加速卡类型",
            options=ACCELERATOR_OPTIONS,
            default=ACCELERATOR_OPTIONS,
            format_func=lambda x: ACCELERATOR_LABELS.get(x, x),
        )
        st.session_state.selected_accelerators = selected_accs

        run_id_filter = st.text_input("Run ID 模糊搜索")

    render_dashboard(run_id_filter)


def render_dashboard(run_id_filter: str):
    st.markdown(
        """
        <h1 style="margin-bottom: 0.2em;">
            📊 综合仪表板
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            margin-top: 0.5em;
            margin-bottom: 1.5em;
            max-width: 80%;
            font-size: 1.3em;
            line-height: 1.6;
        ">
            <strong>InfiniBench Dashboard</strong> 用于统一展示
            <strong>通信（NCCL / 集合通信）</strong>、
            <strong>训练（Training / 分布式训练）</strong>、
            <strong>推理（直接推理 / 服务性能）</strong>、
            <strong>算子（核心算子性能）</strong>、
            <strong>硬件（内存带宽 / 缓存性能）</strong>
            等 AI 加速卡性能测试结果。
            <br/>
            测试框架输出 <code>JSON</code>（环境 / 配置 / 标量指标） +
            <code>CSV</code>（曲线 / 时序数据），
            Dashboard 自动加载并支持多次运行的
            <strong>性能对比</strong>、<strong>趋势分析</strong> 与
            <strong>可视化展示</strong>。
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        runs = st.session_state.data_loader.list_test_runs()
        ci_summaries = st.session_state.data_loader.load_summaries()

        selected_accs = st.session_state.get("selected_accelerators", [])
        if selected_accs:
            runs = [
                r
                for r in runs
                if set(r.get("accelerator_types", [])) & set(selected_accs)
            ]

        if run_id_filter:
            runs = [r for r in runs if run_id_filter in r.get("run_id", "")]

        render_test_run_stats(runs, ci_summaries, selected_accs)
        render_ci_stats(ci_summaries)
        render_latest_results(runs)
        render_dispatcher_summary(ci_summaries)
        render_ci_detailed_table(ci_summaries)
        render_failure_details(ci_summaries)

    except Exception as e:
        st.error(f"Dashboard 加载失败: {e}")


def render_test_run_stats(runs, ci_summaries, selected_accs):
    """Render test run statistics"""
    total = len(runs)
    success = sum(1 for r in runs if r.get("success"))
    fail = total - success

    # Category statistics
    comm_runs = [r for r in runs if r.get("testcase", "").startswith("comm")]
    infer_runs = [r for r in runs if r.get("testcase", "").startswith("infer")]
    train_runs = [r for r in runs if r.get("testcase", "").startswith("train")]

    ops_runs, hw_runs = [], []
    for r in runs:
        p = str(r.get("path", "")).replace("\\", "/").lower()
        tc = (r.get("testcase", "") or "").lower()
        if "/operators/" in p or tc.startswith(("operator", "operators", "ops")):
            ops_runs.append(r)
        if "/hardware/" in p or tc.startswith("hardware"):
            hw_runs.append(r)

    st.markdown("### 📝 测试运行统计")
    st.caption("*基于当前筛选条件的测试运行记录*")

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    col1.metric("总运行数", total)
    col2.metric("成功率", f"{(success/total*100):.1f}%" if total > 0 else "-")
    col3.metric("通信", len(comm_runs))
    col4.metric("推理", len(infer_runs))
    col5.metric("训练", len(train_runs))
    col6.metric("算子", len(ops_runs))
    col7.metric("硬件", len(hw_runs))
    col8.metric("CI记录", len(ci_summaries))

    with st.expander("📈 详细统计"):
        st.caption(f"失败测试数：{fail}")
        st.caption(f"当前筛选：加速卡={','.join(selected_accs) or '全部'}")
    st.divider()


def render_ci_stats(ci_summaries):
    """Render CI statistics"""
    if not ci_summaries:
        return

    total_test_cases = sum(s.get("total_tests", 0) for s in ci_summaries)
    total_success = sum(s.get("successful_tests", 0) for s in ci_summaries)
    total_failed = sum(s.get("failed_tests", 0) for s in ci_summaries)
    total_runs = len(ci_summaries)
    avg_success_rate = (
        (total_success / total_test_cases * 100) if total_test_cases > 0 else 0
    )

    recent_summaries = ci_summaries[:10]
    recent_success = sum(s.get("successful_tests", 0) for s in recent_summaries)
    recent_total = sum(s.get("total_tests", 0) for s in recent_summaries)
    recent_rate = (recent_success / recent_total * 100) if recent_total > 0 else 0

    st.markdown("### 📈 CI运行统计")
    st.caption("*基于Dispatcher汇总的历史CI运行记录*")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("CI运行次数", total_runs)
    col2.metric("测试用例总数", f"{total_test_cases:,}")
    col3.metric("通过用例", f"{total_success:,}")
    col4.metric("失败用例", f"{total_failed:,}")
    col5.metric("平均成功率", f"{avg_success_rate:.1f}%")
    col6.metric(
        "最近10次", f"{recent_rate:.1f}%", delta=f"{recent_rate - avg_success_rate:.1f}%"
    )

    render_daily_trend(ci_summaries)
    st.divider()


def render_daily_trend(ci_summaries):
    """Render daily trend chart"""
    daily_stats = {}
    for s in ci_summaries:
        dt = parse_timestamp(s.get("timestamp", ""))
        if not dt:
            continue
        date_key = dt.strftime("%Y-%m-%d")
        daily_stats[date_key] = {
            "total": daily_stats.get(date_key, {}).get("total", 0)
            + s.get("total_tests", 0),
            "success": daily_stats.get(date_key, {}).get("success", 0)
            + s.get("successful_tests", 0),
        }

    if not daily_stats:
        return

    dates = sorted(daily_stats.keys())
    totals = [daily_stats[d]["total"] for d in dates]
    successes = [daily_stats[d]["success"] for d in dates]
    failures = [totals[i] - successes[i] for i in range(len(dates))]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=totals, name="总测试数", marker_color="lightblue"))
    fig.add_trace(go.Bar(x=dates, y=successes, name="成功", marker_color="lightgreen"))
    fig.add_trace(go.Bar(x=dates, y=failures, name="失败", marker_color="lightcoral"))

    fig.update_layout(
        title="每日测试用例分布趋势",
        barmode="group",
        xaxis_title="日期",
        yaxis_title="测试用例数",
        template="plotly_white",
        height=400,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_latest_results(runs):
    """Render latest test results"""
    st.markdown("### 🚀 最新测试结果")

    def get_latest_by_type(runs, test_type):
        filtered = [r for r in runs if r.get("testcase", "").startswith(test_type)]
        return filtered[0] if filtered else None

    categories = [
        {
            "name": "通信",
            "icon": "🔗",
            "page": "pages/communication.py",
            "run": get_latest_by_type(runs, "comm"),
        },
        {
            "name": "推理",
            "icon": "🚀",
            "page": "pages/inference.py",
            "run": get_latest_by_type(runs, "infer"),
        },
        {
            "name": "算子",
            "icon": "⚡",
            "page": "pages/operator.py",
            "run": get_latest_by_type(runs, "operator"),
        },
        {
            "name": "训练",
            "icon": "🏋️",
            "page": "pages/training.py",
            "run": get_latest_by_type(runs, "train"),
        },
        {
            "name": "硬件",
            "icon": "🔧",
            "page": "pages/hardware.py",
            "run": get_latest_by_type(runs, "hardware"),
        },
    ]

    cols = st.columns(5)
    for idx, cat in enumerate(categories):
        with cols[idx]:
            st.markdown(f"#### {cat['icon']} {cat['name']}（最新）")
            if cat["run"]:
                run = cat["run"]
                st.write(f"- testcase: `{run.get('testcase', '')}`")
                st.write(f"- time: {run.get('time', '')}")
                st.write(f"- status: {'✅' if run.get('success') else '❌'}")
                if st.button(
                    f"查看详情 →", key=f"btn_{cat['name']}", use_container_width=True
                ):
                    st.switch_page(cat["page"])
            else:
                st.info("暂无结果")
    st.divider()


def render_dispatcher_summary(ci_summaries):
    """Render dispatcher summary records"""
    if not ci_summaries:
        return

    st.markdown("### 🧾 Dispatcher 汇总记录")
    st.caption("*每次CI运行的原始汇总文件记录*")

    rows = []
    for s in ci_summaries[:15]:
        rows.append(
            {
                "时间": format_time(s.get("timestamp", "")),
                "总测试数": s.get("total_tests", 0),
                "成功": s.get("successful_tests", 0),
                "失败": s.get("failed_tests", 0),
                "成功率": f"{(s.get('successful_tests', 0)/s.get('total_tests', 1)*100):.1f}%"
                if s.get("total_tests", 0) > 0
                else "-",
                "文件": s.get("file", ""),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.divider()


def render_ci_detailed_table(ci_summaries):
    """Render CI detailed records table"""
    if not ci_summaries:
        st.info("未找到 CI 汇总记录")
        return

    st.markdown("### 📋 CI 详细记录")
    st.caption("*包含Git信息、测试统计和时长的详细记录*")

    rows = []
    for s in ci_summaries[:30]:
        time_str = format_time(s.get("timestamp", ""))
        total = s.get("total_tests", 0)
        success = s.get("successful_tests", 0)
        failed = s.get("failed_tests", 0)
        duration = s.get("duration", s.get("total_duration_seconds", 0))

        # Run ID
        results = s.get("results", [])
        run_id_display = results[0].get("run_id", "-") if results else "-"

        # Git info
        git_info = s.get("git", {})
        short_commit = git_info.get("short_commit", "")
        commit_display = (
            short_commit[:8] if short_commit and short_commit != "unknown" else "本地运行"
        )

        commit_msg = git_info.get("commit_message", "")
        msg_display = (
            commit_msg[:50] + "..."
            if commit_msg and len(commit_msg) > 50
            else (
                commit_msg
                if commit_msg and not commit_msg.startswith("unknown")
                else "-"
            )
        )

        branch = git_info.get("branch", "")
        branch_display = (
            branch if branch and not branch.startswith("unknown") else "local"
        )

        author = git_info.get("commit_author", "")
        author_display = author if author and not author.startswith("unknown") else "-"

        rows.append(
            {
                "时间": time_str,
                "Run ID": run_id_display,
                "Commit": commit_display,
                "提交信息": msg_display,
                "分支": branch_display,
                "作者": author_display,
                "总数": total,
                "✅": success,
                "❌": failed,
                "成功率": f"{(success/total*100):.1f}%" if total > 0 else "-",
                "状态": "✅ 成功" if failed == 0 else "❌ 失败",
                "时长": f"{duration:.1f}s" if duration > 0 else "-",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()


def render_failure_details(ci_summaries):
    """Render failure details"""
    failed_records = []
    for s in ci_summaries:
        failed = s.get("failed_tests", 0)
        if failed == 0:
            continue

        time_str = format_time(s.get("timestamp", ""))
        git_info = s.get("git", {})
        short_commit = git_info.get("short_commit", "")
        commit_display = (
            short_commit[:8] if short_commit and short_commit != "unknown" else "本地运行"
        )

        # Get failure details
        failed_details = s.get("failed_tests_details", [])
        if not failed_details and "results" in s:
            failed_details = [
                {
                    "testcase": r.get("testcase", "unknown"),
                    "run_id": r.get("run_id", "unknown"),
                    "result_code": r.get("result_code", -1),
                    "result_file": r.get("result_file", ""),
                    "error_msg": r.get("error_msg", ""),
                }
                for r in s.get("results", [])
                if r.get("result_code", 0) != 0
            ]

        if failed_details:
            failed_records.append(
                {
                    "时间": time_str,
                    "Commit": commit_display,
                    "失败数": len(failed_details),
                    "失败详情": failed_details,
                }
            )

    if not failed_records:
        return

    st.markdown("### 🔍 失败详情")
    st.caption("点击展开查看失败测试的详细信息")

    for record in failed_records[:15]:
        with st.expander(
            f"📅 {record['时间']} - 失败: {record['失败数']}个测试 (Commit: {record['Commit']})"
        ):
            for i, fail in enumerate(record["失败详情"][:20]):
                st.markdown(f"**{i+1}. {fail.get('testcase', 'unknown')}**")
                st.markdown(f"- Run ID: `{fail.get('run_id', 'unknown')}`")
                st.markdown(f"- Result Code: {fail.get('result_code', -1)}")
                if fail.get("result_file"):
                    st.markdown(f"- Result File: `{fail.get('result_file')}`")
                st.divider()

            if len(record["失败详情"]) > 20:
                st.info(f"还有 {len(record['失败详情']) - 20} 个失败测试未显示")


if __name__ == "__main__":
    main()
