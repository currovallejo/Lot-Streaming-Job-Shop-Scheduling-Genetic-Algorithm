from jobshop import JobShopRandomParams
from genetic.algorithm import GeneticAlgorithm
from genetic.metrics import InMemoryMetrics
from shared.utils import init_logger
from plotting import Plotter


# --------- MAIN DEMO ---------
def main():
    logger = init_logger()

    # Initialize job shop parameters and print them
    problem_params = JobShopRandomParams(config_path="config/jobshop/js_params_01.yaml")
    problem_params.print_jobshop_params()

    # Initialize metrics sink
    metrics_sink = InMemoryMetrics()

    # Run GA with metrics sink
    GA = GeneticAlgorithm(
        problem_params,
        logger,
        config_path="config/genetic_algorithm/ga_config_01.yaml",
        metrics_sink=metrics_sink,
    )
    result = GA.run()

    # Use structured results
    print(f"Best makespan found: {result.makespan}")
    if result.penalty is not None:
        print(f"Penalty: {result.penalty}")

    # Plotting with best solution
    if result.best_solution is not None:
        ops = GA.scheduler.build_operations(result.best_solution)
        plotter = Plotter(problem_params)
        plotter.plot_gantt(ops, save=True, open=True)
        plotter.plot_solution_evolution(result, save=True, open=True)


if __name__ == "__main__":
    main()
