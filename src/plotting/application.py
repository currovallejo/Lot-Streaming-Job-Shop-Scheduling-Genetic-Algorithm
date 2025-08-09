# --- Application layer: orchestrates domain, services, and infrastructure ---

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

# --- External project types (typing only) ---
from ..params import JobShopRandomParams

# --- Internal layers ---
from .domain import GanttConfig
from .mappers import map_dataframe
from .services import build_gantt_figure, build_evolution_figure
from .infra import DefaultFilenamePolicy, HtmlFigureExporter, PngFigureExporter


# --- Gantt orchestration ---


class GanttPlotter:
    """Orchestrates: map DataFrame -> build Plotly figure -> show and/or save HTML."""

    def __init__(self, params: JobShopRandomParams):
        self.params = params
        self.show: bool = True

        self._name_policy = DefaultFilenamePolicy(
            n_machines=params.n_machines,
            n_jobs=params.n_jobs,
            n_lots=params.n_lots,
            seed=params.seed,
            demand=params.demand[0],
            shift_time=params.shift_time,
            shift_constraints=bool(params.shift_constraints),
        )
        self._exporter = HtmlFigureExporter()

    def plot_gantt(
        self,
        df_results: pd.DataFrame,
        save: bool = True,
        open: bool = True,
    ) -> None:
        ts = datetime.now()
        ops = map_dataframe(df_results)
        cfg = GanttConfig(shift_time=self.params.shift_time, auto_open=open)
        fig = build_gantt_figure(ops, cfg)

        if save:
            base = self._name_policy.name_for_gantt(ts)
            self._exporter.export_html(fig, base, auto_open=open)
        elif open:
            fig.show()


# --- Fitness evolution orchestration ---


class SolutionEvolutionPlotter:
    """Orchestrates: build Matplotlib figure -> export PNG."""

    def __init__(self, params: JobShopRandomParams):
        self.params = params
        self.show: bool = True

        self._name_policy = DefaultFilenamePolicy(
            n_machines=len(params.machines),
            n_jobs=len(params.jobs),
            n_lots=len(params.lots),
            seed=params.seed,
            demand=params.demand[0],
            shift_time=params.shift_time,
            shift_constraints=bool(params.shift_constraints),
        )
        self._exporter = PngFigureExporter()

    def plot_evolution(
        self,
        best_fitness_history: Sequence[float],
        save: bool = True,
        open: bool = True,
    ) -> Optional[Path]:
        ts = datetime.now()
        fig = build_evolution_figure(best_fitness_history)
        saved_path: Optional[Path] = None
        if save:
            base = self._name_policy.name_for_evolution(ts)
            saved_path = self._exporter.export_png(fig, base)
        if self.show:
            plt.show()
        return saved_path


# --- Unified application ---


class Plotter:
    """
    Provides a single entry point for plotting, delegating to the Gantt and
    fitness evolution plotters.
    """

    def __init__(self, params: JobShopRandomParams):
        self.params = params
        self.gantt_plotter = GanttPlotter(params)
        self.evolution_plotter = SolutionEvolutionPlotter(params)

    def plot_gantt(
        self,
        df_results: pd.DataFrame,
        save: bool = True,
        open: bool = True,
    ):
        return self.gantt_plotter.plot_gantt(df_results, save, open)

    def plot_solution_evolution(
        self,
        best_fitness_history: Sequence[float],
        save: bool = True,
        open: bool = True,
    ) -> Optional[Path]:
        return self.evolution_plotter.plot_evolution(best_fitness_history, save, open)
