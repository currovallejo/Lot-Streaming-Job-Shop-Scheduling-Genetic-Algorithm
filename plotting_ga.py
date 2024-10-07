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

# --------- OTHER PYTHON FILES USED ---------

# --------- PLOTTING ---------


def plot_gantt(
    df_results, params, show=True, version=0, shifts=False, seq_dep_setup=False
):
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


# --------- PLOT SOLUTION EVOLUTION ---------
def plot_solution_evolution(
    fitness_scores, show=True, save=False, name="solution_evolution"
):
    generations = list(range(1, len(fitness_scores) + 1))
    avg_fitness = [np.mean(scores) for scores in fitness_scores]
    best_fitness = [np.min(scores) for scores in fitness_scores]

    def plot_fitness(generations, fitness, label, color, marker, ylabel, title):
        plt.figure()  # Use automatic figure size

        # Plot the blue line connecting the points
        plt.plot(generations, fitness, color='blue', label=label, linestyle='-')

        # Plot the red points
        plt.scatter(generations, fitness, color='red')

        # Labeling and titles
        plt.xlabel("Generaciones", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.title(title, fontsize=18)

        # No grid
        plt.grid(False)

        # Legend
        plt.legend(fontsize=14)

        # # Custom ticks for x-axis
        # plt.gca().xaxis.set_major_locator(MultipleLocator(100))
        # plt.gca().xaxis.set_minor_locator(MultipleLocator(20))
        # plt.gca().tick_params(axis="x", which="minor", length=4, width=1)
        # plt.gca().tick_params(axis="x", which="major", length=7, width=1.5)

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
