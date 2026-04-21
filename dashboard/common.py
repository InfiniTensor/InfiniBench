#!/usr/bin/env python3
"""Common utilities for dashboard pages."""

import streamlit as st
import sys
from pathlib import Path


def init_page(page_title: str, page_icon: str):
    """
    Common page initialization:
    - -Set Streamlit page configuration
    - initialize DataLoader
    - set project path
    """
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Page configuration
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

    # Initialize use_mongodb setting if not exists
    if "use_mongodb" not in st.session_state:
        st.session_state.use_mongodb = False

    # Initialize DataLoader (respect MongoDB setting)
    if "data_loader" not in st.session_state:
        from utils.data_loader import InfiniBenchDataLoader

        st.session_state.data_loader = InfiniBenchDataLoader(
            use_mongodb=st.session_state.use_mongodb,
            fallback_to_files=True,
        )


def show_data_source_info(style: str = "caption", show_detailed: bool = False):
    """
    Display current data source info (MongoDB or file system).

    Args:
        style: Display style - "caption" for pages, "sidebar" for main app sidebar
        show_detailed: Whether to show detailed statistics
    """
    dl = st.session_state.data_loader

    if dl.source_type == "mongodb":
        if style == "sidebar":
            st.success("🟢 **数据源: MongoDB**")
            if show_detailed:
                st.caption("实时CI数据 | 支持完整历史查询")
        else:
            st.caption("🟢 数据源: MongoDB (实时CI数据)")
    else:
        if style == "sidebar":
            st.info(f"📁 **数据源: 文件系统**")
            st.caption(f"结果目录: `{dl.results_dir}`")
            if show_detailed:
                summary_dir = dl.results_dir.parent / "summary_output"
                st.caption(f"摘要目录: `{summary_dir}`")
        else:
            st.caption(f"📁 数据源: 文件系统 ({dl.results_dir})")
