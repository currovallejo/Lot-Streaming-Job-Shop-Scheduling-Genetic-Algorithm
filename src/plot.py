"""
Created on Aug 08 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: plotting.py - problem and solution plotting
"""

# --------- LIBRARIES ---------
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# --------- src/ MODULES ---------
from .params import JobShopRandomParams

# --------- PLOTTING ---------


class GanttPlotter:
    """
    Class to plot Gantt charts for job shop scheduling solutions.
    """

    def __init__(self, problem_params: JobShopRandomParams):
        self.params = problem_params
        self.show = True

    def plot_gantt(self, df_results):
        """
        Generates and saves an .html file where is plotted the gantt of a job shop
        scheduling program.

        Args:
            df_results: dataframe with solution parameters
        """

        # converting all dataframe numbers to int format
        self.df = df_results.astype(int)
        self._add_df_labels()
        self._build_gantt_figure()
        self._save_gantt()

    def _add_df_labels(self):
        """
        Add labels to the dataframe for gantt info.
        """
        self.df = self.df.assign(
            Products="P " + self.df["job"].astype(str),
            Resources="M " + self.df["machine"].astype(str),
            Text="P" + self.df["job"].astype(str) + " - L" + self.df["lot"].astype(str),
        )

    def _build_gantt_figure(self):
        """
        Build the Gantt figure using Plotly.
        """
        # Precompute durations (vectorized)
        df = self.df.copy()
        df["proc_duration"] = df["completion_time"] - df["start_time"]
        df["setup_duration"] = df["start_time"] - df["setup_start_time"]

        # Processing bars
        fig = px.bar(
            df,
            y="Resources",
            x="proc_duration",
            base="start_time",  # start position on the x axis
            color="Products",
            text="Text",
            orientation="h",
            title="Job Shop Schedule with Lot Streaming",
            hover_data={
                "start_time": True,
                "completion_time": True,
                "lot_size": True,
                "Text": False,
                "Resources": False,
            },
        )

        # Setup bars (hatched, white fill)
        df_setup = df[df["setup_duration"] > 0]
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

        # Merge setup traces into main figure
        for tr in fig_setup.data:
            fig.add_trace(tr)

        # Axes and layout
        fig.update_xaxes(
            type="linear", tick0=0, dtick=self.params.shift_time, title="Time"
        )
        fig.update_yaxes(
            categoryorder="array", categoryarray=sorted(df["Resources"].unique())
        )
        fig.update_layout(barmode="overlay")  # or keep default; overlay is fine here

        # Optional: nicer text placement
        fig.update_traces(textposition="inside", cliponaxis=False)
        self.fig = fig

    def _save_gantt(self):
        """
        Save the Gantt figure to an HTML file.
        """
        params = self.params
        date = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_name = (
            f"GA_m{params.n_machines}_j{params.n_jobs}_u{params.n_lots}_s{params.seed}_d{params.demand[0]}_"
            f"shifts_{params.shift_time}_setup_{params.shift_constraints}_{date}.html"
        )

        out_dir = Path(__file__).resolve().parent.parent / "results" / "gantt"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.fig.write_html(out_dir / file_name, auto_open=self.show)
        self.fig.write_html(
            file_name,
            auto_open=self.show,
        )


class SolutionEvolutionPlotter:
    """
    Class to plot the evolution of the solution's best fitness over generations.
    """

    def __init__(self, params: JobShopRandomParams):
        self.params = params
        self.show = True

    def plot_evolution(self, best_fitness_history):
        pass


class Plotter:
    """
    Class to wrap plotting functionalities.
    """

    def __init__(self, params: JobShopRandomParams):
        self.params = params
        self.gantt_plotter = GanttPlotter(params)

    def plot_gantt(self, df_results):
        """
        Plot the Gantt chart for the given results dataframe.

        Args:
            df_results: DataFrame containing scheduling results.
        """
        self.gantt_plotter.plot_gantt(df_results)


# --------- LEGACY PLOT GANTT ---------
def gantt(df_results, params, show=True, version=0, shifts=False, seq_dep_setup=False):
    """
    Generates and saves an .html file where is plotted the gantt of a job shop
    scheduling program

    Args:
        df_results: dataframe with solution parameters
        params: object of class JobShopRandomParams
        demand: dictionary with the demand of each job
        show: wether to open or not the gantt.html file
        version: to keep track of saved files when doing tests

    Returns:

    """
    n_machines, n_jobs, n_lots, seed = (
        len(params.machines),
        len(params.jobs),
        len(params.lots),
        params.seed,
    )
    demand = params.demand[0]

    # converting all dataframe numbers to int format
    df = df_results.astype(int)

    # add "labels" for plotting discrete
    lst_aux = []
    for j in df["job"].tolist():
        lst_aux.append("P %s" % j)
    df["Products"] = lst_aux

    lst_aux.clear()

    for j in df["machine"].tolist():
        lst_aux.append("M %s" % j)
    df["Resources"] = lst_aux

    lst_aux.clear()

    for i, j in enumerate(df["job"].tolist()):
        u = df.loc[i, "lot"]
        lst_aux.append("P%s - L%s" % (j, u))
    df["Text"] = lst_aux

    # length of the bars
    df["delta"] = df["completion_time"] - df["start_time"]
    print(df["delta"])

    # Create a figure with Plotly colorscale
    fig = px.timeline(
        df,
        x_start="start_time",
        x_end="completion_time",
        y="Resources",
        color="Products",
        title="Job Shop Schedule with Lot Streaming",
        text="Text",
        hover_data={
            "start_time": True,
            "completion_time": True,
            "lot_size": True,
            "Text": False,
            "Resources": False,
        },
    )

    # Set the X-axis type to 'linear'
    fig.layout.xaxis.type = "linear"

    fig.update_xaxes(tick0=0, dtick=params.shift_time)

    for j, Bar in enumerate(fig.data):
        # columna de dataframe de filtrado
        filt = df["Products"] == Bar.name

        # filtrado de la columna delta
        Bar.x = df[filt]["delta"].tolist()

    # length of the setup bars
    df["delta_s"] = df["start_time"] - df["setup_start_time"]

    fig_s = px.timeline(
        df,
        x_start="setup_start_time",
        x_end="start_time",
        y="Resources",
    )

    # Set the X-axis type to 'linear'
    fig_s.layout.xaxis.type = "linear"
    fig_s.update_traces(marker_pattern_shape="/")
    fig_s.update_xaxes(tick0=0, dtick=params.shift_time)
    fig_s.update_traces(marker_color="white")

    for j, Bar in enumerate(fig_s.data):

        Bar.x = df["delta_s"].tolist()
        Bar.legendgroup = "Setup"
        Bar.name = "Setup"
        Bar.showlegend = True
        fig.add_trace(Bar)

    file_name = (
        f"GA_m{n_machines}_j{n_jobs}_u{n_lots}_s{seed}_d{demand}_"
        f"shifts_{shifts}_setup_{seq_dep_setup}_no_pmtn{version}.html"
    )

    fig.write_html(
        file_name,
        auto_open=show,
    )


# --------- LEGACY PLOT SOLUTION EVOLUTION ---------
def solution_evolution(
    fitness_scores, show=True, save=False, name="solution_evolution"
):
    generations = list(range(1, len(fitness_scores) + 1))
    avg_fitness = [np.mean(scores) for scores in fitness_scores]
    best_fitness = [np.min(scores) for scores in fitness_scores]

    def plot_fitness(generations, fitness, label, color, marker, ylabel, title):
        plt.figure()  # Use automatic figure size

        # Plot the blue line connecting the points
        plt.plot(generations, fitness, color="blue", label=label, linestyle="-")

        # Plot the red points
        plt.scatter(generations, fitness, color="red")

        # Labeling and titles
        plt.xlabel("Generaciones", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.title(title, fontsize=18)

        # No grid
        plt.grid(False)

        # Legend
        plt.legend(fontsize=14)

        # Adjust layout
        plt.tight_layout()

        # Show or save plot
        if show:
            plt.show()
        if save:
            plt.savefig(f"{name}.png")

    plot_fitness(
        generations,
        best_fitness,
        "Mejor Fitness",
        "tab:blue",
        "o",
        "Fitness",
        "Evolución del Mejor Fitness",
    )
    # plot_fitness(
    #     generations,
    #     avg_fitness,
    #     "Media Fitness",
    #     "tab:red",
    #     "x",
    #     "Fitness Promedio",
    #     "Evolución del Fitness Promedio",
    # )
