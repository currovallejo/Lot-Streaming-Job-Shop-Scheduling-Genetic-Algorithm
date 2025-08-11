"""
Unified plotting interface for Lot Streaming Job Shop Scheduling Problem visualization.

This module provides a single entry point for all plotting operations, delegating
to specialized plotters for Gantt charts and fitness evolution visualization.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

from jobshop import JobShopRandomParams
from domain import ScheduledOperation
from genetic.results import GAResult
from .gantt import plot_and_export_gantt
from .evolution import plot_and_export_evolution


class Plotter:
    """
    Provides a single entry point for plotting, delegating to specialized
    Gantt and fitness evolution plotters.
    """

    def __init__(self, params: JobShopRandomParams):
        """Initialize the Plotter with job shop parameters.
        Args:
            params (JobShopRandomParams): Parameters for the job shop problem.
        """
        self.params = params

    def plot_gantt(
        self,
        ops: Sequence[ScheduledOperation],
        save: bool = True,
        open: bool = True,
    ) -> None:
        """Plot Gantt chart for scheduled operations.
        Args:
            ops (Sequence[ScheduledOperation]): List of scheduled operations to visualize.
            save (bool): Whether to save the figure as HTML. Defaults to True.
            open (bool): Whether to open the saved HTML file in a browser. Defaults to True.
        """
        return plot_and_export_gantt(ops, self.params, save, open)

    def plot_solution_evolution(
        self,
        result: GAResult,
        save: bool = True,
        open: bool = True,
        title: str = "Fitness Evolution",
    ) -> Optional[Path]:
        """Plot fitness evolution from GA results.
        Args:
            result (GAResult): Results from the genetic algorithm run.
            save (bool): Whether to save the figure as PNG. Defaults to True.
            open (bool): Whether to open the saved PNG file. Defaults to True.
            title (str): Title for the plot. Defaults to "Fitness Evolution".
        Returns:
            Optional[Path]: Path to the saved PNG file if saved, otherwise None.
        """
        return plot_and_export_evolution(result, self.params, save, open, title)
