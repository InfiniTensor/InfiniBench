#!/usr/bin/env python3
"""Visualization functions for InfiniBench dashboard."""

import plotly.graph_objects as go
import pandas as pd
from typing import Optional
import streamlit as st


def create_gauge_chart(
    value: float,
    max_value: float,
    title: str,
    color: str = "blue",
    unit: str = "",
    decimals: Optional[int] = None,  # optional
) -> go.Figure:
    """Create a gauge chart for single metric visualization."""

    if decimals is None:
        if value < 10:
            decimals = 2
        elif value < 100:
            decimals = 1
        else:
            decimals = 0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0.05, 0.85]},
            title={"text": title, "font": {"size": 18}},
            number={
                "suffix": f" {unit}",
                "font": {"size": 36},
                "valueformat": f".{decimals}f",
            },
            gauge={
                "axis": {"range": [0, max_value], "tickformat": f".{decimals}f"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, max_value * 0.6], "color": "lightgray"},
                    {"range": [max_value * 0.6, max_value * 0.8], "color": "gray"},
                    {"range": [max_value * 0.8, max_value], "color": "darkgray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": max_value * 0.9,
                },
            },
        )
    )

    fig.update_layout(height=260, margin=dict(t=20, b=0, l=10, r=10))
    return fig


def plot_timeseries_auto(
    df: pd.DataFrame, title: str = "Timeseries", y_log_scale: bool = False
) -> go.Figure:
    """Generic plot for 2-column timeseries CSV."""
    fig = go.Figure()
    if df is None or df.empty or len(df.columns) < 2:
        fig.update_layout(title=f"{title} (no data)")
        return fig

    xcol = df.columns[0]
    ycol = df.columns[1]

    fig.add_trace(go.Scatter(x=df[xcol], y=df[ycol], mode="lines+markers", name=ycol))

    fig.update_layout(
        title=title,
        xaxis_title=str(xcol),
        yaxis_title=str(ycol),
        template="plotly_white",
        height=420,
        hovermode="x unified",
    )
    fig.update_yaxes(rangemode="tozero")
    if y_log_scale:
        fig.update_yaxes(type="log")
    return fig
