#!/usr/bin/env python3
"""Hardware-specific visualization functions for InfiniBench dashboard."""

import plotly.graph_objects as go
import pandas as pd

# Color constants
COLOR_MEMORY = "#2196F3"
COLOR_MEMORY_FILL = "rgba(33, 150, 243, 0.1)"
COLOR_CACHE = "#E91E63"
COLOR_CACHE_FILL = "rgba(233, 30, 99, 0.1)"
COLOR_AVG_LINE = "#9E9E9E"
COLOR_GRID = "rgba(200,200,200,0.3)"

# Layout defaults
_LAYOUT_DEFAULTS = {
    "template": "plotly_white",
    "height": 450,
    "hovermode": "closest",
    "margin": dict(t=60, b=40, l=60, r=30),
    "showlegend": False,
}


def _apply_common_style(fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str):
    """Apply common layout and styling to a hardware figure."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        **_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLOR_GRID)
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor=COLOR_GRID, rangemode="tozero"
    )


def _add_avg_line(fig: go.Figure, y_values: pd.Series):
    """Add average line annotation to figure."""
    avg_val = y_values.mean()
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color=COLOR_AVG_LINE,
        annotation_text=f"Avg: {avg_val:.1f}",
        annotation_position="right",
    )


def create_summary_table_hw(test_result: dict) -> pd.DataFrame:
    """Create summary table for hardware test results."""
    rows = [{"指标": "testcase", "数值": str(test_result.get("testcase", ""))}]

    env = test_result.get("environment", {})
    try:
        acc = env["cluster"][0]["machine"]["accelerators"][0]
        rows += [
            {"指标": "加速卡", "数值": str(acc.get("model", "Unknown"))},
            {"指标": "卡数", "数值": str(acc.get("count", "Unknown"))},
        ]
    except (KeyError, IndexError, TypeError):
        pass

    for m in test_result.get("metrics", []):
        if m.get("type") == "scalar":
            rows.append(
                {
                    "指标": str(m.get("name", "")),
                    "数值": f"{m.get('value', '')} {m.get('unit', '')}".strip(),
                }
            )

    return pd.DataFrame(rows)


def plot_hw_mem_sweep(
    df: pd.DataFrame, title: str = "Memory Sweep", y_log_scale: bool = False
) -> go.Figure:
    """Plot memory sweep bandwidth: x=size_mb, y=bandwidth_gbps."""
    fig = go.Figure()

    if df is None or df.empty:
        fig.update_layout(title=f"{title} (no data)")
        return fig

    xcol, ycol = "size_mb", "bandwidth_gbps"
    if xcol not in df.columns or ycol not in df.columns:
        xcol = df.columns[0]
        ycol = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    fig.add_trace(
        go.Scatter(
            x=df[xcol],
            y=df[ycol],
            mode="lines+markers",
            name="Bandwidth",
            line=dict(color=COLOR_MEMORY, width=2.5, shape="spline"),
            marker=dict(size=8, color=COLOR_MEMORY, line=dict(color="white", width=1)),
            fill="tozeroy",
            fillcolor=COLOR_MEMORY_FILL,
            hovertemplate="<b>%{x}</b> MB<br/>Bandwidth: <b>%{y:.2f}</b> GB/s<extra></extra>",
        )
    )

    _apply_common_style(fig, title, "Size (MB)", "Bandwidth (GB/s)")
    fig.update_xaxes(type="log", range=[0, 3])  # 1MB to 1000MB
    _add_avg_line(fig, df[ycol])

    if y_log_scale:
        fig.update_yaxes(type="log")

    return fig


def plot_hw_cache(
    df: pd.DataFrame, title: str = "Cache Bandwidth", y_log_scale: bool = False
) -> go.Figure:
    """Plot cache bandwidth: x=exec_data or data_set, y=eff_bw."""
    fig = go.Figure()

    if df is None or df.empty:
        fig.update_layout(title=f"{title} (no data)")
        return fig

    # exec_data (L2) preferred over data_set (L1)
    ycol = "eff_bw"
    if "exec_data" in df.columns and ycol in df.columns:
        xcol = "exec_data"
    elif "data_set" in df.columns and ycol in df.columns:
        xcol = "data_set"
    else:
        xcol = df.columns[0]
        ycol = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    fig.add_trace(
        go.Scatter(
            x=df[xcol],
            y=df[ycol],
            mode="lines+markers",
            name="Effective BW",
            line=dict(color=COLOR_CACHE, width=2.5, shape="spline"),
            marker=dict(size=8, color=COLOR_CACHE, line=dict(color="white", width=1)),
            fill="tozeroy",
            fillcolor=COLOR_CACHE_FILL,
            hovertemplate="Data Set: <b>%{x}</b><br/>BW: <b>%{y:.2f}</b> GB/s<extra></extra>",
        )
    )

    _apply_common_style(fig, title, "Data Set", "Effective BW (GB/s)")
    _add_avg_line(fig, df[ycol])

    if y_log_scale:
        fig.update_yaxes(type="log")

    return fig
