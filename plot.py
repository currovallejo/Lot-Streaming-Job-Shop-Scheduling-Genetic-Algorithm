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
import matplotlib.animation as animation

# --------- OTHER PYTHON FILES USED ---------

# --------- PLOTTING ---------


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


def plot_bars(ax, df, params, show=True, save=False, name="bar_plot"):
    """
    Plots a bar chart using matplotlib where Y coordinates are machines and X coordinates are time intervals.
    Bars with the same 'job' number in the df will have the same color.

    Args:
        df: DataFrame with columns 'machine', 'start_time', 'completion_time', and 'job'
        params: object of class JobShopRandomParams
        show: whether to display the plot
        save: whether to save the plot as an image
        name: name of the file to save the plot

    Returns:
        None
    """

    # Generate a color map
    unique_jobs = df["job"].unique()
    colors = plt.cm.get_cmap("Paired", len(unique_jobs))

    # Plot bars
    seen_labels = set()
    for i, row in df.iterrows():
        job_index = np.where(unique_jobs == row["job"])[0][0]
        label = f"Job {int(row['job'])}"
        if label not in seen_labels:
            ax.barh(
                row["machine"],
                row["completion_time"] - row["start_time"],
                left=row["start_time"],
                edgecolor="black",
                color=colors(job_index),
                label=label,
            )
            seen_labels.add(label)
        else:
            ax.barh(
                row["machine"],
                row["completion_time"] - row["setup_start_time"],
                left=row["start_time"],
                edgecolor="black",
                color=colors(job_index),
            )

    for i, row in df.iterrows():
        job_index = np.where(unique_jobs == row["job"])[0][0]
        job_label = f"Job {int(row['job'])}"
        setup_label = "Setup"

        # Plot job bars
        if job_label not in seen_labels:
            ax.barh(
                row["machine"],
                row["completion_time"] - row["start_time"],
                left=row["start_time"],
                edgecolor="black",
                color=colors(job_index),
                label=job_label,
            )
            seen_labels.add(job_label)
        else:
            ax.barh(
                row["machine"],
                row["completion_time"] - row["start_time"],
                left=row["start_time"],
                edgecolor="black",
                color=colors(job_index),
            )

        # Plot setup bars
        if setup_label not in seen_labels:
            ax.barh(
                row["machine"],
                row["start_time"] - row["setup_start_time"],
                left=row["setup_start_time"],
                edgecolor="black",
                color="white",
                label=setup_label,
                hatch="//",
            )
            seen_labels.add(setup_label)
        else:
            ax.barh(
                row["machine"],
                row["start_time"] - row["setup_start_time"],
                left=row["setup_start_time"],
                edgecolor="black",
                color="white",
                hatch="//",
            )

    # Add vertical lines every 480 units
    max_time = df["completion_time"].max()
    for x in range(0, max_time, 480):
        ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.5)

    # Labeling and titles
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Machines", fontsize=14)
    ax.set_title("Job Shop Schedule", fontsize=16)

    # Ensure Y-axis only shows integer numbers
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add legend
    ax.legend(title="Jobs", fontsize=12, title_fontsize=14)

    # Remove grid
    ax.grid(False)

    # # Adjust layout
    # plt.tight_layout()

    # Show or save plot
    if show:
        plt.show()
    if save:
        plt.savefig(f"{name}.png")


def generate_gif(dataframes, params, filename="gantt.gif", fps=1):
    """
    Generates a GIF showing the evolution of the job shop instance using multiple dataframes.

    Args:
        dataframes: List of DataFrames, each representing a different state of the job shop instance.
        params: object of class JobShopRandomParams
        gif_name: Name of the output GIF file.
        fps: Frames per second for the GIF.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10000)

    def update(frame):
        ax.clear()
        plot_bars(ax, dataframes[frame], params, show=False)
        ax.set_title("Lot Streaming Job Shop Scheduling Genetic Algorithm Evolution")
        ax.set_xlim(0, 10000)

    ani = animation.FuncAnimation(fig, update, frames=len(dataframes), repeat=False)
    writergif = animation.PillowWriter(fps=5)
    ani.save(filename, writer=writergif)
    plt.close()


def generate_gif2(dataframes, filename="gantt.gif", fps=5):
    """
    Generates a GIF showing the evolution of the job shop instance using multiple dataframes.

    Args:
        dataframes: List of DataFrames, each representing a different state of the job shop instance.
        params: object of class JobShopRandomParams
        gif_name: Name of the output GIF file.
        fps: Frames per second for the GIF.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate the maximum time value across all dataframes
    max_time = max(df["completion_time"].max() for df in dataframes)

    # Set the X-axis limits to keep it consistent across all frames
    ax.set_xlim(0, max_time)

    # Flatten the list of DataFrames into a list of rows
    rows = [(i, row) for i, df in enumerate(dataframes) for _, row in df.iterrows()]
    print('hay ', len(rows), 'filas en la lista flat de los dfs')

    # Extract the last len(dataframes[0]) rows
    last_rows = rows[-len(dataframes[0]):]

    # Append the last rows multiple times to hold them
    rows.extend(last_rows * 2)

    # Generate a color map
    unique_jobs = dataframes[0]["job"].unique()
    colors = plt.cm.get_cmap("Paired", len(unique_jobs))

    def update(frame):
        print(frame)
        ax.clear()
        start_index = frame
        end_index = frame + len(dataframes[0])
        current_rows = rows[start_index:end_index]

        seen_labels = set()
        for df_index, row in current_rows:
            job_index = np.where(unique_jobs == row["job"])[0][0]
            job_label = f"Job {int(row['job'])}"
            setup_label = "Setup"

            # Plot job bars
            if job_label not in seen_labels:
                ax.barh(
                    row["machine"],
                    row["completion_time"] - row["start_time"],
                    left=row["start_time"],
                    edgecolor="black",
                    color=colors(job_index),
                    label=job_label,
                )
                seen_labels.add(job_label)
            else:
                ax.barh(
                    row["machine"],
                    row["completion_time"] - row["start_time"],
                    left=row["start_time"],
                    edgecolor="black",
                    color=colors(job_index),
                )

            # Plot setup bars
            if setup_label not in seen_labels:
                ax.barh(
                    row["machine"],
                    row["start_time"] - row["setup_start_time"],
                    left=row["setup_start_time"],
                    edgecolor="black",
                    color="white",
                    label=setup_label,
                    hatch="//",
                )
                seen_labels.add(setup_label)
            else:
                ax.barh(
                    row["machine"],
                    row["start_time"] - row["setup_start_time"],
                    left=row["setup_start_time"],
                    edgecolor="black",
                    color="white",
                    hatch="//",
                )

        # Set the X-axis limits to keep it consistent across all frames
        ax.set_xlim(0, max_time + 4000)

        # Add vertical lines every 480 units
        for x in range(0, max_time, 480):
            ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.5)

        # Calculate the maximum completion time for the current rows
        current_max_time = max(row["completion_time"] for _, row in current_rows)

        # Add text annotation for the maximum completion time
        ax.text(
            max_time +100, 0.25, f"Makespan: {current_max_time:.2f}",
            verticalalignment='center', horizontalalignment='left',
            fontsize=12, color='red', zorder=10
        )

        # Labeling and titles
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Machines", fontsize=14)
        ax.set_title("Lot Streaming Job Shop Scheduling", fontsize=16)

        # Ensure Y-axis only shows integer numbers
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add legend
        # Sort legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(
            zip(labels, handles), key=lambda x: (x[0] != "Setup", x[0])
        )
        sorted_labels, sorted_handles = zip(*sorted_handles_labels)
        ax.legend(
            sorted_handles, sorted_labels, title="Jobs", fontsize=12, title_fontsize=14
        )

        # Remove grid
        ax.grid(False)

        # Adjust layout manually
        plt.tight_layout()


    ani = animation.FuncAnimation(
        fig, update, frames=len(rows) - len(dataframes[0]), repeat=False
    )
    writergif = animation.PillowWriter(fps=fps)
    ani.save(filename, writer=writergif)

    plt.close(fig)


# --------- PLOT SOLUTION EVOLUTION ---------
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
