# --------- LIBRARIES ---------
import time

# --------- OTHER PYTHON FILES USED ---------
import src.params as params
import src.decoder as decoder
from src.genetic_algorithm import GeneticAlgorithm
from src.utils import timed, init_logger
from src.plot import Plotter


# --------- MAIN FUNCTIONALITY ---------
logger = init_logger()
problem_params = params.JobShopRandomParams(
    config_path="config/jobshop/js_params_01.yaml"
)
problem_params.print_jobshop_params(save_to_excel=False)

GA = GeneticAlgorithm(
    problem_params, logger, config_path="config/genetic_algorithm/ga_config_01.yaml"
)
makespan, best_fitness_history, best_individual = GA.run()

# Print results
print("Best makespan found:", makespan)
print("Best fitness history:", best_fitness_history)

# Plotting results
df_results = GA.js_decoder.get_schedule_df_from_solution(best_individual)
plotter = Plotter(problem_params)
plotter.plot_gantt(df_results)

# --------- GENETIC ALGORITHM (CHANGE FOR DIFFERENT GA CONFIGURATIONS) ---------
# Genetic algorithm
# start = time.time()
# current_fitness, best_individual = run(
#     my_params,
#     population_size=100,
#     num_generations=100,
#     plotting=True,
# )
# end = time.time()
# print("Time elapsed: ", end - start, "seconds")
# print("Current fitness: ", current_fitness)
# if my_params.shift_constraints:
#     js_decoder = decoder.JobShopDecoder(my_params)
#     makespan, penalty, y, c, chromosome_mod = js_decoder.decode(best_individual)
#     print("Makespan: ", makespan)
#     print("Penalty: ", penalty)
# print(best_individual)
# print(chromosome_mod)
