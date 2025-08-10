# from jobshop as jobshop
from jobshop import JobShopRandomParams
from genetic.algorithm import GeneticAlgorithm
from shared.utils import init_logger
from plotting import Plotter


# --------- MAIN DEMO ---------
logger = init_logger()
problem_params = JobShopRandomParams(
    config_path="config/jobshop/js_params_01.yaml"
)
problem_params.print_jobshop_params(save_to_excel=False)

GA = GeneticAlgorithm(
    problem_params, logger, config_path="config/genetic_algorithm/ga_config_01.yaml"
)
makespan, best_fitness_history, best_individual = GA.run()

# Print results
print("Best makespan found:", makespan)
if problem_params.shift_constraints:
    penalty = GA.scheduler.decode(best_individual)[1]
    print("Penalty:", penalty)
print("Best fitness history:", best_fitness_history)

# Plotting results
ops = GA.scheduler.build_operations(best_individual)
plotter = Plotter(problem_params)
plotter.plot_gantt(ops, save=True, open=True)
plotter.plot_solution_evolution(best_fitness_history, save=True, open=True)
