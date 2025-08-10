def test_chromosome_generator(problem_params, chromosome):
    """
    Check if the chromosome generator produces a valid chromosome.
    """

    # Check if the chromosome has the expected structure
    assert isinstance(chromosome, tuple), "Chromosome should be a tuple"

    # Check if the lot sizes are non-negative integers
    lot_sizes = chromosome[0]
    assert all(
        size >= 0 for size in lot_sizes
    ), "Lot sizes should be non-negative integers"

    # Check if the chromosome left-hand side has the expected number of lots
    total_lots = problem_params.n_jobs * problem_params.n_lots
    assert (
        len(chromosome[0]) == total_lots
    ), "Chromosome length should match number of total lots"

    # Check if the chromosome right-hand side has the expected number of pairs (jobs, lots)
    total_ops = problem_params.n_lots * sum(
        len(problem_params.seq[j]) for j in problem_params.jobs
    )

    assert (
        len(chromosome[1]) == total_ops
    ), "Chromosome right-hand side should match number of total lots"
