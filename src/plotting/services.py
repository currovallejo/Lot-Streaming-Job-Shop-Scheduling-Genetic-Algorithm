# --- Application services: build figures (no I/O here) ---

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..domain import ScheduledOperation


# --- Private label helpers (presentation-only) ---
def _product_label(job: int) -> str:
    return f"P {job}"


def _resource_label(machine: int) -> str:
    return f"M {machine}"


def _text_label(job: int, lot: int) -> str:
    return f"P{job} - L{lot}"


# --- Gantt figure ---


def build_gantt_figure(
    ops: Sequence[ScheduledOperation],
    shift_time: int,
    title: str = "Job Shop Schedule with Lot Streaming",
) -> go.Figure:
    """
    Create a Plotly Gantt-like bar chart from domain operations.
    Pure function: returns a figure, does not write files or open windows.
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


# --- Evolution figure ---


def build_evolution_figure(
    best_fitness_history: Sequence[float],
    title: str = "Best Fitness Evolution",
) -> plt.Figure:
    """
    Create a Matplotlib figure for the best fitness across generations.
    Args:
        best_fitness_history: Sequence of best fitness values per generation.
        title: Title for the figure.
    Returns:
        fig: Matplotlib figure object.
    """
    generations = list(range(1, len(best_fitness_history) + 1))

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(generations, best_fitness_history, linestyle="-", label="Best Fitness")
    ax.scatter(generations, best_fitness_history)

    ax.set_xlabel("Generations", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(False)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig
