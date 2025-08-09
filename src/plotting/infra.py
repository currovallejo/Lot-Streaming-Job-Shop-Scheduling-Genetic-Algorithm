# --- Infrastructure adapters: filenames and persistence (I/O) ---

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure


# --- Filename policy ---


@dataclass
class DefaultFilenamePolicy:
    """Builds base filenames from problem parameters."""

    n_machines: int
    n_jobs: int
    n_lots: int
    seed: int
    demand: int
    shift_time: int
    shift_constraints: bool

    # --- Base name builder ---
    def _base_name(self, prefix: str, now: datetime) -> str:
        stamp = now.strftime("%Y%m%d_%H%M%S")
        return (
            f"{prefix}_m{self.n_machines}"
            f"_j{self.n_jobs}"
            f"_u{self.n_lots}"
            f"_s{self.seed}"
            f"_d{self.demand}"
            f"_shifts_{self.shift_time}"
            f"_setup_{self.shift_constraints}"
            f"_{stamp}"
        )

    # --- Gantt base name ---
    def name_for_gantt(self, now: datetime) -> str:
        return self._base_name("schedule", now)

    # --- Evolution base name ---
    def name_for_evolution(self, now: datetime) -> str:
        return self._base_name("evolution", now)


# --- Plotly exporter (HTML) ---


class HtmlFigureExporter:
    """Writes Plotly figures to HTML in results/gantt/ folder."""

    def __init__(self):
        # Fixed output directory relative to repo root
        self.output_dir = Path(__file__).resolve().parents[2] / "results" / "schedule"

    def export_html(self, fig: go.Figure, base_name: str, auto_open: bool) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{base_name}.html"
        fig.write_html(path, auto_open=auto_open)
        return path


# --- Matplotlib exporter (PNG) ---


class PngFigureExporter:
    """Writes Matplotlib figures to PNG in results/fitness_evolution/ folder."""

    dpi: int = 150

    def __init__(self):
        # Fixed output directory relative to repo root
        self.output_dir = (
            Path(__file__).resolve().parents[2] / "results" / "fitness_evolution"
        )

    def export_png(self, fig: MplFigure, base_name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{base_name}.png"
        fig.savefig(path, dpi=self.dpi)
        return path
