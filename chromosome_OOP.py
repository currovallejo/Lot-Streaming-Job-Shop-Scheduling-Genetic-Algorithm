"""
Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
Website: https://franciscovallejo.pro

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: chromosome.py - migration of chromosome_generator.py and decoder.py to a class

*FAILED* - GA runs slower
"""

# --------- LIBRARIES ---------
import random
import pandas as pd
import numpy as np

# --------- OTHER PYTHON FILES USED ---------
import params as parameters
import plot


class Chromosome:
    """
    Represents a chromosome in the job-shop lot streaming scheduling problem.
    Handles the creation, decoding, and evaluation of chromosomes based on the problem's parameters.
    """

    def __init__(self, params: parameters.JobShopRandomParams):
        """
        Initializes the Chromosome object with the parameters required to generate a chromosome.

        Args:
            params: An instance of JobShopRandomParams, containing jobs, lots, and sequences.
        """
        self.params = params
        self.n_jobs = len(params.jobs)
        self.n_lots = len(params.lots)
        self.n_machines = len(params.machines)

    def generate(self):
        """
        Generates a random chromosome.

        Attributes added:
            self.chromosome_lhs: A numpy array (left-hand side) of numeric values in
            [0, 1] representing the fractions of demand for each lot
            self.chromosome_rhs: a shuffled list (right-hand side) of tuples (job, lot)
        """
        params = self.params
        # Generate chromosome left-hand side (numeric values)
        chromosome_lhs = np.array(
            [random.random() for job in params.jobs for lot in params.lots]
        )

        # Generate chromosome right-hand side (sublots)
        chromosome_rhs = [
            (job, lot)
            for job in params.jobs
            for lot in params.lots
            for machine in params.seq[job]
        ]

        # Shuffle right-hand side to randomize sublots
        random.shuffle(chromosome_rhs)

        self.chromosome_lhs = chromosome_lhs
        self.chromosome_rhs = chromosome_rhs

    def distribute_demand(self):
        """
        Distributes the total demand into parts based on the generated chromosome fractions
        such that the sum of the parts equals the total demand for each job.

        Attributes added:
            self.chromosome_lhs_m: numpy array with the demand distributed into lots (LHS of the chromosome modified)
        """
        params = self.params
        if self.chromosome_lhs is None:
            raise ValueError(
                "Chromosome LHS not generated yet. Call the generate method first."
            )

        chromosome_lhs_m = np.copy(self.chromosome_lhs)

        # Distribute demand for each job
        for job in params.jobs:
            total = np.sum(
                chromosome_lhs_m[self.n_lots * job : self.n_lots * (job + 1)]
            )

            if total != 0:  # Avoid division by zero
                chromosome_lhs_m[self.n_lots * job : self.n_lots * (job + 1)] = (
                    chromosome_lhs_m[self.n_lots * job : self.n_lots * (job + 1)]
                    / total
                )

                for lot in params.lots:
                    chromosome_lhs_m[self.n_lots * job + lot] = int(
                        chromosome_lhs_m[self.n_lots * job + lot] * params.demand[job]
                    )

                total_preliminary = sum(
                    chromosome_lhs_m[self.n_lots * job : self.n_lots * (job + 1)]
                )
                residual = params.demand[job] - total_preliminary

                # Distribute the residual if there is any
                if residual > 0:
                    for lot in params.lots:
                        chromosome_lhs_m[self.n_lots * job + lot] += 1
                        residual -= 1
                        if residual == 0:
                            break
            else:
                # Handle case where total is 0
                chromosome_lhs_m[self.n_lots * job : self.n_lots * (job + 1)] = (
                    int(params.demand[job]) / self.n_lots
                )
                chromosome_lhs_m[self.n_lots * job + self.n_lots - 1] = params.demand[
                    job
                ] - sum(
                    chromosome_lhs_m[self.n_lots * job : self.n_lots * (job + 1) - 1]
                )

        self.chromosome_lhs_m = chromosome_lhs_m

    def decode(self, shifts, seq_dep_setup):
        """
        Decodes a chromosome to a solution

        Returns:
            y: numpy array with setup start times of all lots
            c: numpy array with completion times of all lots
            makespan: integer with makespan of the solution
        """
        params = self.params

        # Base functions for active scheduling - level 1
        def is_empty_machine() -> bool:
            if seq_dep_setup:
                if precedences[current_machine] == [
                    -1
                ]:  # if is first lot in the machine
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

        # Base functions for active scheduling - level 2
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

        def is_big_lotsize() -> bool:
            if seq_dep_setup:
                lot_size_time = (
                    params.sd_setup[
                        current_machine,
                        current_job + 1,
                        precedences[current_machine][-1][0] + 1,
                    ]
                    + params.p_times[current_machine, current_job]
                    * self.chromosome_lhs_m[self.n_lots * current_job + current_lot]
                )
            else:
                lot_size_time = (
                    params.setup[current_machine, current_job]
                    + params.p_times[current_machine, current_job]
                    * self.chromosome_lhs_m[self.n_lots * current_job + current_lot]
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
                    * self.chromosome_lhs_m[self.n_lots * current_job + current_lot]
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
                    * self.chromosome_lhs_m[self.n_lots * current_job + current_lot]
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
                    * self.chromosome_lhs_m[self.n_lots * current_job + current_lot]
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
                    * self.chromosome_lhs_m[self.n_lots * current_job + current_lot]
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
                        y[current_machine, current_job, current_lot] = (
                            params.shift_time * (last_shift + 1)
                        )

                elif not is_first_machine() and is_empty_machine():
                    last_completion = completion_in_previous_machine()
                    if fit_within_previous_shift():
                        y[current_machine, current_job, current_lot] = last_completion
                    else:
                        y[current_machine, current_job, current_lot] = (
                            params.shift_time
                            * (last_completion // params.shift_time + 1)
                        )

                elif not is_first_machine() and not is_empty_machine():
                    last_completion_current_machine = completion_in_current_machine()
                    last_completion_previous_machine = completion_in_previous_machine()

                    if (
                        last_completion_previous_machine
                        >= last_completion_current_machine
                    ):
                        if fit_within_previous_shift():
                            y[current_machine, current_job, current_lot] = (
                                last_completion_previous_machine
                            )
                        else:
                            y[current_machine, current_job, current_lot] = (
                                params.shift_time
                                * (
                                    last_completion_previous_machine
                                    // params.shift_time
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
                                * (
                                    last_completion_current_machine // params.shift_time
                                    + 1
                                )
                            )

        # Decodification of chromosome
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
        y = np.full((self.n_machines, self.n_jobs, self.n_lots), 0)  # setup start time
        c = np.full((self.n_machines, self.n_jobs, self.n_lots), 0)  # completion time

        # Schedule the jobs and get the makespan
        makespan = 0
        penalty_coefficient = 1e2
        max_lot_penalty = 0
        penalty = 0

        for i, job_lot in enumerate(self.chromosome_rhs):  # For each lot
            if (
                self.chromosome_lhs_m[self.n_lots * job_lot[0] + job_lot[1]] != 0
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
                        * self.chromosome_lhs_m[self.n_lots * current_job + current_lot]
                    )
                else:
                    c[current_machine, current_job, current_lot] = (
                        y[current_machine, current_job, current_lot]
                        + params.setup[current_machine, current_job]
                        + params.p_times[current_machine, current_job]
                        * self.chromosome_lhs_m[self.n_lots * current_job + current_lot]
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

        self.makespan = makespan
        self.penalty = penalty
        self.y = y
        self.c = c

    def get_chromosome_start_times(self):
        """
        Calculates start times of all lots

        Args:
            chromosome: dataframe with solution parameters
            params: object of class JobShopRandomParams
            c: completion time of each lot

        Returns:
            x: numpy array with start time of each lot
        """
        if self.c is None:
            raise ValueError(
                "Completion time 'c' cannot be None, call decode method first."
            )

        params = self.params
        triple_mju = {
            (m, j, u)
            for m in range(self.n_machines)
            for j in range(self.n_jobs)
            for u in range(self.n_lots)
        }

        x = np.full((self.n_machines, self.n_jobs, self.n_lots), 0)  # start time
        for m, j, u in triple_mju:
            if self.c[m, j, u] > 0:
                x[m, j, u] = (
                    self.c[m, j, u]
                    - params.p_times[m, j] * self.chromosome_lhs_m[self.n_lots * j + u]
                )

        self.x = x

    def build_chromosome_df(self):
        """
        Builds a dataframe to show chromosome parameters decoded

        Args:
            y: numpy array with setup start times of all lots
            c: numpy array with completion times of all lots
            x: numpy array with start time of each lot

        Returns:
            df: dataframme with solution results
        """
        y = self.y
        x = self.x
        c = self.c
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
        df["lot_size"] = df.apply(
            lambda row: self.chromosome_lhs_m[(row["job"] * num_lots) + row["lot"]],
            axis=1,
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


def main():
    my_params = parameters.JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=4)
    my_params.printParams(sequence_dependent=True)

    my_chromosome = Chromosome(my_params)
    my_chromosome.chromosome_lhs = np.array(
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
    )
    my_chromosome.chromosome_rhs = [
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
    ]

    my_chromosome.distribute_demand()
    my_chromosome.decode(shifts=True, seq_dep_setup=True)
    my_chromosome.get_chromosome_start_times()
    results_df = my_chromosome.build_chromosome_df()
    print("Makespan:", my_chromosome.makespan)
    print("Penalty:", my_chromosome.penalty)
    plot.gantt(results_df, my_params, show=True, version=0, shifts=True, seq_dep_setup=True)


if __name__ == "__main__":
    main()
