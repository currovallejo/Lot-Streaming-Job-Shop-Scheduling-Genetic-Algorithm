"""
Gantt chart visualization for job shop scheduling solutions.

This module provides functions for generating and exporting Gantt charts from
scheduled operations using Plotly for interactive visualization.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog
"""

from __future__ import annotations
from datetime import datetime
from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from jobshop import JobShopRandomParams
from domain import ScheduledOperation
from .infra import DefaultFilenamePolicy, HtmlFigureExporter


# --- Private label helpers (presentation-only) ---
def _product_label(job: int) -> str:
    return f"P {job}"


def _resource_label(machine: int) -> str:
    return f"M {machine}"


def _text_label(job: int, lot: int) -> str:
    return f"P{job} - L{lot}"


def build_gantt_figure(
    ops: Sequence[ScheduledOperation],
    shift_time: int,
    title: str = "Job Shop Schedule with Lot Streaming",
) -> go.Figure:
    """
    Create a Plotly Gantt-like bar chart from domain operations.
    Args:
        ops (Sequence[ScheduledOperation]): List of scheduled operations to visualize.
        shift_time (int): Shift time for the job shop.
        title (str): Title for the Gantt chart.
    Returns:
        go.Figure: A Plotly figure object representing the Gantt chart.
    """
    if not ops:
        return go.Figure(layout_title_text=title)

    # --- View-model dataframe for plotting (durations from domain) ---
    df = pd.DataFrame(
        {
            "Products": [_product_label(op.id.job) for op in ops],
            "Resources": [_resource_label(op.id.machine) for op in ops],
            "Text": [_text_label(op.id.job, op.id.lot) for op in ops],
            "start_time": [op.time.start for op in ops],
            "completion_time": [op.time.completion for op in ops],
            "setup_start_time": [op.time.setup_start for op in ops],
            "lot_size": [op.lot_size for op in ops],
            "proc_duration": [op.time.proc_duration for op in ops],
            "setup_duration": [op.time.setup_duration for op in ops],
        }
    )

    # --- Processing bars ---
    fig = px.bar(
        df,
        y="Resources",
        x="proc_duration",
        base="start_time",
        color="Products",
        text="Text",
        orientation="h",
        title=title,
        hover_data={
            "start_time": True,
            "proc_duration": False,
            "completion_time": True,
            "lot_size": True,
            "Text": False,
            "Resources": False,
        },
    )

    # --- Setup bars (hatched, white fill) ---
    df_setup = df[df["setup_duration"] > 0]
    if not df_setup.empty:
        fig_setup = px.bar(
            df_setup,
            y="Resources",
            x="setup_duration",
            base="setup_start_time",
            orientation="h",
        )
        fig_setup.update_traces(
            marker_color="white",
            marker_pattern_shape="/",
            name="Setup",
            legendgroup="Setup",
            showlegend=True,
        )
        for tr in fig_setup.data:
            fig.add_trace(tr)

    # --- Axes and layout ---
    fig.update_xaxes(type="linear", tick0=0, dtick=shift_time, title="Time")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=sorted(df["Resources"].unique()),
        title=None,
    )
    fig.update_layout(barmode="overlay")
    fig.update_traces(textposition="inside", cliponaxis=False)

    return fig


def plot_and_export_gantt(
    ops: Sequence[ScheduledOperation],
    params: JobShopRandomParams,
    save: bool = True,
    open: bool = True,
) -> None:
    """
    Create and optionally save/display a Gantt chart.

    Args:
        ops: Scheduled operations to visualize
        params: Job shop parameters for naming and configuration
        save: Whether to save as HTML file
        open: Whether to open in browser (if saving) or show interactive plot
    """
    fig = build_gantt_figure(ops, params.shift_time)

    if save:
        name_policy = DefaultFilenamePolicy(
            n_machines=params.n_machines,
            n_jobs=params.n_jobs,
            n_lots=params.n_lots,
            seed=params.seed,
            demand=params.demand[0],
            shift_time=params.shift_time,
            shift_constraints=bool(params.shift_constraints),
        )
        exporter = HtmlFigureExporter()
        base = name_policy.name_for_gantt(datetime.now())
        exporter.export_html(fig, base, auto_open=open)
    elif open:
        fig.show()
