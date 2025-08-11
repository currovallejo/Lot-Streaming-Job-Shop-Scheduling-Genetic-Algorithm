"""
Fitness evolution visualization for genetic algorithm results.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from jobshop import JobShopRandomParams
from genetic.results import GAResult
from .infra import DefaultFilenamePolicy, PngFigureExporter


def plot_fitness_evolution(
    result: GAResult, title: str = "Best Fitness Evolution"
) -> Figure:
    """Create a Matplotlib figure for fitness evolution from GA results."""
    generations = list(range(1, len(result.best_ever_per_gen) + 1))

    fig = plt.figure()
    ax = fig.gca()

    # Plot both best per generation and best ever
    ax.plot(
        generations,
        result.best_fitness_per_gen,
        linestyle="-",
        label="Best Fitness per Generation",
        alpha=0.7,
    )
    ax.plot(
        generations,
        result.best_ever_per_gen,
        linestyle="-",
        label="Best Fitness Ever",
        linewidth=2,
    )
    ax.scatter(generations, result.best_ever_per_gen, s=20)

    ax.set_xlabel("Generations", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.grid(False)

    # Enhanced title with makespan and optional penalty
    enhanced_title = f"{title} (Makespan: {result.makespan}"
    if result.penalty is not None:
        enhanced_title += f", Penalty: {result.penalty}"
    enhanced_title += ")"

    ax.set_title(enhanced_title, fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()

    return fig


def plot_and_export_evolution(
    result: GAResult,
    params: JobShopRandomParams,
    save: bool = True,
    show: bool = True,
    title: str = "Fitness Evolution",
) -> Optional[Path]:
    """
    Create and optionally save/display a fitness evolution plot.

    Args:
        result: GA results containing fitness evolution data
        params: Job shop parameters for naming and configuration
        save: Whether to save as PNG file
        show: Whether to display the plot
        title: Title for the plot

    Returns:
        Path to saved file if saved, otherwise None
    """
    fig = plot_fitness_evolution(result, title)
    saved_path: Optional[Path] = None

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
        exporter = PngFigureExporter()
        base = name_policy.name_for_evolution(datetime.now())
        saved_path = exporter.export_png(fig, base)

    if show:
        plt.show()

    return saved_path
