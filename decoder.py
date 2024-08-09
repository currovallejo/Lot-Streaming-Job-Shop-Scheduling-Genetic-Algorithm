"""
Created on Aug 05 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: decoder.py - decodification of the chromosome to the solution
"""

#--------- LIBRARIES ---------
import numpy as np
import pandas as pd

#--------- OTHER PYTHON FILES USED ---------
import chromosome_generator
import params
import plotting_ga

#--------- DECODER ---------

def decode_chromosome(chromosome, params):
    """
    Decodes a chromosome to a solution

    Args:
        chromosome: numpy array with the chromosome
        params: object of class JobShopRandomParams

    Returns:
        y: numpy array with setup start times of all lots
        c: numpy array with completion times of all lots
        makespan: integer with makespan of the solution
    """
    def distribute_demand():
        """
        Distributes a total demand into parts based on the given fractions
        such that the sum of the parts equals the total demand.

        Args:

        Returns:
            chromosome_lhs: numpy array with the demand distributed into lots
        """
        # Calculate lot sizes for each job
        for job in range(chromosome_lhs.shape[0]):
            
            total = np.sum(chromosome_lhs[job])
            if total != 0:  # Avoid division by zero
                chromosome_lhs[job] = chromosome_lhs[job] / total

                for lot in range(chromosome_lhs.shape[1]):
                    chromosome_lhs[job, lot] = int(chromosome_lhs[job, lot]*params.demand[job])
                total_preliminary = sum(chromosome_lhs[job])
                residual = params.demand[job] - total_preliminary

                if residual>0:
                    indices = np.arange(len(chromosome_lhs[job]))

                    for i in indices:
                        chromosome_lhs[job, i] += 1
                        residual -= 1
                        if residual == 0:
                            break
            
            else:
                for lot in chromosome_lhs[job]:
                    chromosome[job][lot] = int(params.demand[job]) / len(chromosome_lhs[job])
                chromosome_lhs[job][-1] = (params.demand[job]) - sum(chromosome_lhs[job][:-1])
        
        return chromosome_lhs
    
    # Initialize variables
    chromosome_lhs = chromosome[0]
    print(chromosome_lhs)
    chromosome_lhs = distribute_demand()
    print(chromosome_lhs)
    chromosome_rhs = chromosome[1]
    n_jobs = len(params.jobs)
    n_machines = len(params.machines)
    n_lots = len(params.lots)

    # Do a dictionary to track route of each lot
    routes = {
        (job, lot): params.seq[job][:]
        for job in params.jobs 
        for lot in params.lots
    }

    # Dictionary to track precedence in scheduling
    precedences = {}
    for machine in params.machines:
        precedences[machine] = []
    
    # Arrays to store times of all sublots
    y = np.full((n_machines, n_jobs, n_lots), 0) # setup start time
    c = np.full((n_machines, n_jobs, n_lots), 0) # completion time

    # Schedule the jobs and get the makespan
    makespan = 0
    for i, job_lot in enumerate(chromosome_rhs): # For each lot
        if chromosome_lhs[job_lot[0],job_lot[1]]!=0: # If the lot is not empty
            current_lot = job_lot[1]
            current_job = job_lot[0]
            current_machine = routes[(current_job, current_lot)][0]

            # Calculate the start time and completion of the lot
            if precedences[current_machine] == []: # if is first lot in the machine
                empty_machine = True
            else:
                empty_machine = False
            
            if current_machine == params.seq[current_job][0]: # if is first machine in the job route
                first_machine = True
            else:
                first_machine = False
            
            if empty_machine and first_machine:
                y[current_machine, current_job, current_lot] = 0

            elif empty_machine and not first_machine:
                previous_machine = params.seq[current_job][params.seq[current_job].index(current_machine)-1] # previous machine in the job route
                y[current_machine, current_job, current_lot] = c[previous_machine, current_job, current_lot]

            elif not empty_machine and first_machine:
                predecessor = precedences[current_machine][-1] # predecessor in the current machine
                y[current_machine, current_job, current_lot] = c[current_machine, predecessor[0], predecessor[1]]
            
            elif not empty_machine and not first_machine:
                previous_machine = params.seq[current_job][params.seq[current_job].index(current_machine)-1]
                c_previous_machine = c[previous_machine, current_job, current_lot]
                predecessor = precedences[current_machine][-1]
                c_predecessor = c[current_machine, predecessor[0], predecessor[1]]

                y[current_machine, current_job, current_lot] = max(c_previous_machine, c_predecessor)
            
            # Calculate the completion time of the lot
            c[current_machine, current_job, current_lot] = y[current_machine, current_job, current_lot] + params.setup[current_machine, current_job] + params.p_times[current_machine, current_job]*chromosome_lhs[current_job, current_lot]
        
            # Update makespan
            if c[current_machine, current_job, current_lot] > makespan:
                makespan = c[current_machine, current_job, current_lot]
            
            # Update precedences
            precedences[current_machine].append((current_job, current_lot))

            # Update lot route
            routes[(current_job, current_lot)].pop(0)
    
    return makespan, y, c

def get_chromosome_start_times(chromosome, params, c):
    """
    Calculates start times of all lots 

    Args:
        chromosome: dataframe with solution parameters
        params: object of class JobShopRandomParams
        c: completion time of each lot

    Returns:
        x: numpy array with start time of each lot
    """
    n_machines, n_jobs, n_lots = len(params.machines), len(params.jobs), len(params.lots)
    triple_mju = {(m, j, u) 
                for m in range(n_machines)
                for j in range(n_jobs)
                for u in range(n_lots)
            }
    
    x = np.full((n_machines, n_jobs, n_lots), 0) # setup start time
    for m,j,u in triple_mju:
        if c[m,j,u]>0:
            x[m,j,u] = c[m,j,u] - params.p_times[m,j]*chromosome[0][j,u]
    
    return x

def build_chromosome_results_df(chromosome, y, x, c):
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
    df = pd.DataFrame(s_start_time_2d, columns=['setup_start_time'])

    # Add additional columns
    df['start_time'] = start_time_2d
    df['completion_time'] = completion_time_2d

    # Generate additional columns for machine, job, and lot
    df['machine'] = np.repeat(np.arange(num_machines), num_jobs * num_lots)
    df['job'] = np.tile(np.repeat(np.arange(num_jobs), num_lots), num_machines)
    df['lot'] = np.tile(np.arange(num_lots), num_machines * num_jobs)

    # Add the lot_size column based on the job and lot indices
    df['lot_size'] = df.apply(lambda row: chromosome[0][row['job'], row['lot']], axis=1)

    # Reorder columns if needed
    df = df[['machine', 'job', 'lot', 'setup_start_time', 'start_time', 'completion_time', 'lot_size']]

    # Filter out rows where completion_time is 0
    df_filtered = df[df['completion_time'] != 0]
    df_filtered = df_filtered.reset_index(drop=True)

    # Display the DataFrame
    print(df_filtered)

    return df_filtered

def main():
    # Generate a random chromosome
    my_params = params.JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=4)
    my_params.printParams()
    demand = {i: 50 for i in range(0, 11)}
    chromosome = chromosome_generator.generate_chromosome(my_params, demand)

    # Decode the chromosome
    makespan, y, c = decode_chromosome(chromosome, my_params)
    print("makespan: \n", makespan)
    print('setup start times: \n', y)
    print('completion times: \n' ,c)

    # Get start time of each lot
    x = get_chromosome_start_times(chromosome, my_params, c)
    print("Start times: \n", x)
    
    # Build dataframe with chromosome solution results
    df_results = build_chromosome_results_df(chromosome, y, x, c)

    # Plot gantt
    plotting_ga.plot_gantt(df_results, my_params, demand) 

if __name__ == "__main__":
    main()


    
