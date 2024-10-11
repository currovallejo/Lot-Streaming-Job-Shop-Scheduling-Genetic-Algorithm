"""
Created on Aug 05 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: decoder.py - decodification of the chromosome to the solution
"""

# --------- LIBRARIES ---------
import numpy as np
import pandas as pd

# --------- OTHER PYTHON FILES USED ---------
import params
import plot

# --------- DECODER ---------


def decode_chromosome(chromosome, params, shifts=False, seq_dep_setup=False):
    """
    Decodes a chromosome to a solution

    Args:
        chromosome: numpy array with the chromosome
        params: object of class JobShopRandomParams
        shifts: boolean to indicate if shift constraints are considered
        seq_dep_setup: boolean to indicate if setup times are sequence dependent

    Returns:
        y: numpy array with setup start times of all lots
        c: numpy array with completion times of all lots
        makespan: integer with makespan of the solution
    """

    # Functions for active scheduling
    def distribute_demand():
        """
        Distributes a total demand into parts based on the given fractions
        such that the sum of the parts equals the total demand.

        Args:

        Returns:
            chromosome_lhs_m: numpy array with the demand distributed into lots
                (LHS of the chromosome modified)
        """
        chromosome_lhs_m = np.copy(chromosome[0])
        for job in params.jobs:
            total = np.sum(chromosome_lhs_m[n_lots * job : n_lots * (job + 1)])

            if total != 0:  # Avoid division by zero
                chromosome_lhs_m[n_lots * job : n_lots * (job + 1)] = (
                    chromosome_lhs_m[n_lots * job : n_lots * (job + 1)] / total
                )

                for lot in params.lots:
                    chromosome_lhs_m[n_lots * job + lot] = int(
                        chromosome_lhs_m[n_lots * job + lot] * params.demand[job]
                    )
                total_preliminary = sum(
                    chromosome_lhs_m[n_lots * job : n_lots * (job + 1)]
                )
                residual = params.demand[job] - total_preliminary

                if residual > 0:
                    for lot in params.lots:
                        chromosome_lhs_m[n_lots * job + lot] += 1
                        residual -= 1
                        if residual == 0:
                            break

            else:
                chromosome_lhs_m[n_lots * job : n_lots * (job + 1)] = (
                    int(params.demand[job]) / n_lots
                )
                chromosome_lhs_m[n_lots * job + n_lots - 1] = (
                    params.demand[job]
                ) - sum(chromosome_lhs_m[n_lots * job : n_lots * (job)])

        return chromosome_lhs_m

    def is_empty_machine() -> bool:
        if seq_dep_setup:
            if precedences[current_machine] == [-1]:  # if is first lot in the machine
                precedences[current_machine]
                empty_machine = True
            else:
                empty_machine = False
        else:
            if precedences[current_machine] == []:  # if is first lot in the machine
                empty_machine = True
            else:
                empty_machine = False
        return empty_machine

    def is_first_machine() -> bool:
        if (
            current_machine == params.seq[current_job][0]
        ):  # if is first machine in the job route
            first_machine = True
        else:
            first_machine = False
        return first_machine

    def completion_in_previous_machine():
        previous_machine = params.seq[current_job][
            params.seq[current_job].index(current_machine) - 1
        ]  # previous machine in the job route
        next_start_time = c[previous_machine, current_job, current_lot]
        return next_start_time

    def completion_in_current_machine():
        predecessor = precedences[current_machine][
            -1
        ]  # predecessor in the current machine
        next_start_time = c[current_machine, predecessor[0], predecessor[1]]
        return next_start_time

    def lot_start_time():
        if is_first_machine() and is_empty_machine():
            y[current_machine, current_job, current_lot] = 0

        elif is_first_machine() and not is_empty_machine():
            y[current_machine, current_job, current_lot] = (
                completion_in_current_machine()
            )

        elif not is_first_machine() and is_empty_machine():
            y[current_machine, current_job, current_lot] = (
                completion_in_previous_machine()
            )

        elif not is_first_machine() and not is_empty_machine():
            y[current_machine, current_job, current_lot] = max(
                completion_in_current_machine(), completion_in_previous_machine()
            )

    # Functions for shift constraints

    def is_big_lotsize() -> bool:
        if seq_dep_setup:
            lot_size_time = (
                params.sd_setup[
                    current_machine,
                    current_job + 1,
                    precedences[current_machine][-1][0] + 1,
                ]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
            )
        else:
            lot_size_time = (
                params.setup[current_machine, current_job]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
            )
        if lot_size_time > params.shift_time:
            return True
        else:
            return False

    def fit_within_predecessor_shift():
        if seq_dep_setup:
            if (
                completion_in_current_machine() % params.shift_time
                + params.sd_setup[
                    current_machine,
                    current_job + 1,
                    precedences[current_machine][-1][0] + 1,
                ]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
                <= params.shift_time
            ):
                return True
            else:
                return False
        else:
            if (
                completion_in_current_machine() % params.shift_time
                + params.setup[current_machine, current_job]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
                <= params.shift_time
            ):
                return True
            else:
                return False

    def fit_within_previous_shift():
        if seq_dep_setup:
            if (
                completion_in_previous_machine() % params.shift_time
                + params.sd_setup[
                    current_machine,
                    current_job + 1,
                    precedences[current_machine][-1][0] + 1,
                ]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
                <= params.shift_time
            ):
                return True
            else:
                return False
        else:
            if (
                completion_in_previous_machine() % params.shift_time
                + params.setup[current_machine, current_job]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
                <= params.shift_time
            ):
                return True
            else:
                return False

    def lot_start_time_with_shifts():
        if is_big_lotsize():
            lot_start_time()
        else:
            if is_first_machine() and is_empty_machine():
                y[current_machine, current_job, current_lot] = 0

            elif is_first_machine() and not is_empty_machine():
                last_completion = completion_in_current_machine()
                if fit_within_predecessor_shift():
                    y[current_machine, current_job, current_lot] = last_completion
                else:
                    last_shift = last_completion // params.shift_time
                    y[current_machine, current_job, current_lot] = params.shift_time * (
                        last_shift + 1
                    )

            elif not is_first_machine() and is_empty_machine():
                last_completion = completion_in_previous_machine()
                if fit_within_previous_shift():
                    y[current_machine, current_job, current_lot] = last_completion
                else:
                    y[current_machine, current_job, current_lot] = params.shift_time * (
                        last_completion // params.shift_time + 1
                    )

            elif not is_first_machine() and not is_empty_machine():
                last_completion_current_machine = completion_in_current_machine()
                last_completion_previous_machine = completion_in_previous_machine()

                if last_completion_previous_machine >= last_completion_current_machine:
                    if fit_within_previous_shift():
                        y[current_machine, current_job, current_lot] = (
                            last_completion_previous_machine
                        )
                    else:
                        y[current_machine, current_job, current_lot] = (
                            params.shift_time
                            * (
                                last_completion_previous_machine // params.shift_time
                                + 1
                            )
                        )

                else:
                    if fit_within_predecessor_shift():
                        y[current_machine, current_job, current_lot] = (
                            last_completion_current_machine
                        )
                    else:
                        y[current_machine, current_job, current_lot] = (
                            params.shift_time
                            * (last_completion_current_machine // params.shift_time + 1)
                        )

    # Initialize variables
    n_jobs = len(params.jobs)
    n_machines = len(params.machines)
    n_lots = len(params.lots)
    chromosome_lhs = chromosome[0]
    chromosome_rhs = chromosome[1]

    # decode left hand side of the chromosome if it is a float
    if chromosome_lhs[0] is np.float64:
        chromosome_lhs_m = distribute_demand()
    else:
        chromosome_lhs_m = np.copy(chromosome_lhs)

    chromosome_mod = [chromosome_lhs_m, chromosome_rhs]

    # Do a dictionary to track route of each lot
    routes = {
        (job, lot): params.seq[job][:] for job in params.jobs for lot in params.lots
    }

    # Dictionary to track precedence in scheduling
    precedences = {}
    for machine in params.machines:
        if seq_dep_setup:
            precedences[machine] = [(-1, 0)]
        else:
            precedences[machine] = []

    # Arrays to store times of all sublots
    y = np.full((n_machines, n_jobs, n_lots), 0)  # setup start time
    c = np.full((n_machines, n_jobs, n_lots), 0)  # completion time

    # Schedule the jobs and get the makespan
    makespan = 0
    penalty_coefficient = 1e2
    max_lot_penalty = 0
    penalty = 0
    for i, job_lot in enumerate(chromosome_rhs):  # For each lot
        if (
            chromosome_lhs_m[n_lots * job_lot[0] + job_lot[1]] != 0
        ):  # If the lot is not empty
            current_job = job_lot[0]
            current_lot = job_lot[1]
            current_machine = routes[(current_job, current_lot)][0]

            # Calculate the start time and completion of the lot
            if shifts:
                lot_start_time_with_shifts()
            else:
                lot_start_time()

            # Calculate the completion time of the lot
            if seq_dep_setup:
                c[current_machine, current_job, current_lot] = (
                    y[current_machine, current_job, current_lot]
                    + params.sd_setup[
                        current_machine,
                        current_job + 1,
                        precedences[current_machine][-1][0] + 1,
                    ]
                    + params.p_times[current_machine, current_job]
                    * chromosome_lhs_m[n_lots * current_job + current_lot]
                )
            else:
                c[current_machine, current_job, current_lot] = (
                    y[current_machine, current_job, current_lot]
                    + params.setup[current_machine, current_job]
                    + params.p_times[current_machine, current_job]
                    * chromosome_lhs_m[n_lots * current_job + current_lot]
                )

            # Update makespan
            if c[current_machine, current_job, current_lot] > makespan:
                makespan = c[current_machine, current_job, current_lot]

            if is_big_lotsize():
                lot_penalty = (
                    c[current_machine, current_job, current_lot] % params.shift_time
                )
                if lot_penalty > max_lot_penalty:
                    max_lot_penalty = lot_penalty
                    penalty = penalty_coefficient * max_lot_penalty

            # Update precedences
            precedences[current_machine].append((current_job, current_lot))

            # Update lot route
            routes[(current_job, current_lot)].pop(0)

    return makespan, penalty, y, c, chromosome_mod


def get_chromosome_start_times(chromosome_mod, params, c):
    """
    Calculates start times of all lots

    Args:
        chromosome: dataframe with solution parameters
        params: object of class JobShopRandomParams
        c: completion time of each lot

    Returns:
        x: numpy array with start time of each lot
    """
    chromosome_lhs_m = chromosome_mod[0]
    n_machines, n_jobs, n_lots = (
        len(params.machines),
        len(params.jobs),
        len(params.lots),
    )
    triple_mju = {
        (m, j, u)
        for m in range(n_machines)
        for j in range(n_jobs)
        for u in range(n_lots)
    }

    x = np.full((n_machines, n_jobs, n_lots), 0)  # start time
    for m, j, u in triple_mju:
        if c[m, j, u] > 0:
            x[m, j, u] = (
                c[m, j, u] - params.p_times[m, j] * chromosome_lhs_m[n_lots * j + u]
            )

    return x


def build_chromosome_results_df(chromosome_mod, y, x, c):
    """
    Builds a dataframe to show chromosome solution results

    Args:
        y: numpy array with setup start times of all lots
        c: numpy array with completion times of all lots
        x: numpy array with start time of each lot

    Returns:
        df: dataframme with solution results
    """
    # Reshape the 3D array to a 2D array
    # Shape will be (num_machines * num_jobs * num_lots, 1)
    num_machines, num_jobs, num_lots = y.shape
    s_start_time_2d = y.reshape(num_machines * num_jobs * num_lots, 1)
    start_time_2d = x.reshape(num_machines * num_jobs * num_lots, 1)
    completion_time_2d = c.reshape(num_machines * num_jobs * num_lots, 1)

    # Create a DataFrame
    df = pd.DataFrame(s_start_time_2d, columns=["setup_start_time"])

    # Add additional columns
    df["start_time"] = start_time_2d
    df["completion_time"] = completion_time_2d

    # Generate additional columns for machine, job, and lot
    df["machine"] = np.repeat(np.arange(num_machines), num_jobs * num_lots)
    df["job"] = np.tile(np.repeat(np.arange(num_jobs), num_lots), num_machines)
    df["lot"] = np.tile(np.arange(num_lots), num_machines * num_jobs)

    # Add the lot_size column based on the job and lot indices
    chromosome_lhs_m = chromosome_mod[0]
    df["lot_size"] = df.apply(
        lambda row: chromosome_lhs_m[(row["job"] * num_lots) + row["lot"]], axis=1
    )

    # Reorder columns if needed
    df = df[
        [
            "machine",
            "job",
            "lot",
            "setup_start_time",
            "start_time",
            "completion_time",
            "lot_size",
        ]
    ]

    # Filter out rows where completion_time is 0
    df_filtered = df[df["completion_time"] != 0]
    df_filtered = df_filtered.reset_index(drop=True)

    # Display the DataFrame
    print(df_filtered)

    return df_filtered


def get_dataframe_results(chromosome, params, shifts=False, seq_dep_setup=False):
    """
    Get the results of a chromosome in a dataframe

    Args:
        chromosome: numpy array with the chromosome
        params: object of class JobShopRandomParams
        shifts: boolean to indicate if shift constraints are considered
        seq_dep_setup: boolean to indicate if setup times are sequence dependent

    Returns:
        df_results: dataframe with the results of the chromosome
    """
    # Decode the chromosome
    _, _, y, c, chromosome_mod = decode_chromosome(
        chromosome, params, shifts, seq_dep_setup
    )

    # Get start time of each lot
    x = get_chromosome_start_times(chromosome_mod, params, c)

    # Build dataframe with chromosome solution results
    df_results = build_chromosome_results_df(chromosome_mod, y, x, c)

    return df_results


def main():
    # Generate Job Shop Random Parameters
    my_params = params.JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=4)
    my_params.printParams(sequence_dependent=True)
    demand = {i: 100 for i in range(0, 11)}

    # Example chromosome (could be random generated but for testing purposes is fixed)
    chromosome = [
        np.array(
            [
                0.33893225,
                0.39426217,
                0.567908,
                0.67242174,
                0.37249561,
                0.85690449,
                0.62416912,
                0.51083418,
                0.21353647,
            ]
        ),
        [
            (1, 2),
            (2, 2),
            (0, 0),
            (0, 2),
            (0, 0),
            (1, 1),
            (1, 0),
            (0, 0),
            (1, 1),
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 1),
            (1, 2),
            (0, 1),
            (2, 0),
            (0, 2),
            (0, 1),
        ],
    ]

    # Decode the chromosome
    makespan, penalty, y, c, chromosome_mod = decode_chromosome(
        chromosome, my_params, shifts=True, seq_dep_setup=True
    )
    print("makespan: \n", makespan)
    print("penalty: \n", penalty)
    print("setup start times: \n", y)
    print("completion times: \n", c)

    # Get start time of each lot
    x = get_chromosome_start_times(chromosome_mod, my_params, c)
    print("Start times: \n", x)

    # Build dataframe with chromosome solution results
    df_results = build_chromosome_results_df(chromosome_mod, y, x, c)

    # Plot gantt
    plot.gantt(df_results, my_params, demand)
    print("nice job")


if __name__ == "__main__":
    main()
