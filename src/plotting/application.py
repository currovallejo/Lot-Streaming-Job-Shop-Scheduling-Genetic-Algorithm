"""
Plotting application for Lot Streaming Job Shop Scheduling Problem visualization.

This module provides a unified plotting interface for visualizing genetic algorithm
solutions including Gantt charts for job schedules and fitness evolution plots.
It orchestrates figure generation, export functionality (HTML/PNG), and provides
automated filename policies for saving visualization outputs.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
import matplotlib.pyplot as plt

from jobshop import JobShopRandomParams
from domain import ScheduledOperation
from .services import build_gantt_figure, build_evolution_figure
from .infra import DefaultFilenamePolicy, HtmlFigureExporter, PngFigureExporter


class GanttPlotter:
    """Orchestrates: map DataFrame -> build Plotly figure -> show and/or save HTML."""

    def __init__(self, params: JobShopRandomParams):
        """Initialize the Gantt plotter with job shop parameters
        Args:
            params (JobShopRandomParams): Parameters for the job shop problem.
        """
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
        ops: Sequence[ScheduledOperation],
        save: bool = True,
        open: bool = True,
    ) -> None:
        """Plot Gantt chart for the given operations.
        Args:
            ops (Sequence[ScheduledOperation]): List of scheduled operations to visualize.
            save (bool): Whether to save the figure as HTML. Defaults to True.
            open (bool): Whether to open the saved HTML file in a browser. Defaults to True.
        """
        ts = datetime.now()
        shift_time = self.params.shift_time
        fig = build_gantt_figure(ops, shift_time)

        if save:
            base = self._name_policy.name_for_gantt(ts)
            self._exporter.export_html(fig, base, auto_open=open)
        elif open:
            fig.show()


# --- Fitness evolution orchestration ---


class SolutionEvolutionPlotter:
    """Orchestrates: build Matplotlib figure -> export PNG."""

    def __init__(self, params: JobShopRandomParams):
        """Initialize the solution evolution plotter with job shop parameters.
        Args:
            params (JobShopRandomParams): Parameters for the job shop problem.
        """
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
        """Plot the evolution of the best fitness values of each generation.
        Args:
            best_fitness_history (Sequence[float]): List of best fitness value of each generation.
            save (bool): Whether to save the figure as PNG. Defaults to True.
            open (bool): Whether to open the saved PNG file. Defaults to True.
        Returns:
            Optional[Path]: Path to the saved PNG file if saved, otherwise None.
        """
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
        """Initialize the Plotter with job shop parameters.
        Args:
            params (JobShopRandomParams): Parameters for the job shop problem.
        """
        self.params = params
        self.gantt_plotter = GanttPlotter(params)
        self.evolution_plotter = SolutionEvolutionPlotter(params)

    def plot_gantt(
        self,
        ops: Sequence[ScheduledOperation],
        save: bool = True,
        open: bool = True,
    ) -> None:
        return self.gantt_plotter.plot_gantt(ops, save, open)

    def plot_solution_evolution(
        self,
        best_fitness_history: Sequence[float],
        save: bool = True,
        open: bool = True,
    ) -> Optional[Path]:
        return self.evolution_plotter.plot_evolution(best_fitness_history, save, open)
