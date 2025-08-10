import pytest
from src import jobshop
from src import ChromosomeGenerator


@pytest.fixture(scope="session")
def problem_params():
    return jobshop.JobShopRandomParams("config/jobshop/js_params_01.yaml")


@pytest.fixture(scope="session")
def chromosome(problem_params):
    return ChromosomeGenerator(problem_params).generate()
