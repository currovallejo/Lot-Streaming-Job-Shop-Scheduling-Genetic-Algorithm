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
        # category_orders={'Machine': sorted(df['Machine'].unique())},
        # labels={'y':'Machine', 'x':'Job'},
        # range_x=[0,max(df['makespan'])]
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
    # print(df['delta_s'])

    fig_s = px.timeline(
        df,
        x_start="setup_start_time",
        x_end="start_time",
        y="Resources",
        # category_orders={'Machine': sorted(df['Machine'].unique())},
        # labels={'y':'Machine', 'x':'Job'},
        # range_x=[0,max(df['makespan'])]
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
