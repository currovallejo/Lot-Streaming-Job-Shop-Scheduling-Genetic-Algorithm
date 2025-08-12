

# Lot Streaming Job Shop Scheduling Genetic Algorithm

![Test](https://github.com/currovallejo/Lot-Streaming-Job-Shop-Scheduling-Genetic-Algorithm/actions/workflows/test.yaml/badge.svg)
![Ruff](https://github.com/currovallejo/Lot-Streaming-Job-Shop-Scheduling-Genetic-Algorithm/actions/workflows/ruff.yaml/badge.svg)
![Mypy](https://github.com/currovallejo/Lot-Streaming-Job-Shop-Scheduling-Genetic-Algorithm/actions/workflows/mypy.yaml/badge.svg)

This repository implements a genetic algorithm to solve the **Lot Streaming Job Shop Scheduling Problem (LSJSP)**, , with support for **sequence-dependent setup times** and **shift constraints**. It includes modules for generating random problem instances, decoding chromosomes into semi-active schedules, and visualizing results.

| ![newplot](https://github.com/user-attachments/assets/a19958ea-16c8-4115-a7dc-7d9ade8d5ded) | 
|:--:| 
| *LSJSP Scheduling with shift constraints and sequence dependent setup times* |

## 🚀 Key features

- 🎲 Random generation of Job Shop parameters from YAML config files
- 🧬 Genetic algorithm built with **DEAP** including multiple crossover and mutation operators
- 🗺️ Decoder that supports sequence-dependent or independent setup times and optional shift constraints
- 📦 Lot streaming with consistent sizes. Lot sizes are equal for all operations along the routing process of a job, but the size of each lot is variable and determined by the algorithm
- 📊 Plotting utilities for Gantt charts and fitness evolution stored under `results/`


## 📋 Requirements
- Python **3.11.4**
- [Poetry](https://python-poetry.org/) for dependency management
  - **⚠️ IMPORTANT:** Poetry is strictly required for this project. The application uses Poetry's package structure for imports and won't function with standard pip installations or other package managers. All scripts depend on Poetry's environment configuration.

## ⚡ Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/currovallejo/Lot-Streaming-Job-Shop-Scheduling-Genetic-Algorithm.git
cd Lot-Streaming-Job-Shop-Scheduling-Genetic-Algorithm
```

2. **Install Poetry** (Windows PowerShell command)

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

*(For macOS/Linux, see [Poetry installation docs](https://python-poetry.org/docs/#installation))*

3. **Configure Poetry to store the virtual environment inside the project**

```bash
poetry config virtualenvs.in-project true
```

4. **Add Poetry to your system PATH**
   Make sure the Poetry `bin` directory is in your `PATH` so the `poetry` command is recognized.
   Verify with:

```bash
poetry --version
```

5. **Install all dependencies (including development tools)**

```bash
poetry install --with dev
```

6. **Run the application or tests**

```bash
poetry run python main.py     # Run main script
poetry run pytest -q          # Run tests
```

## Using Jupyter Notebooks with Poetry 📓

You can explore or demonstrate the project interactively using Jupyter Notebooks, fully integrated with your Poetry-managed environment.

1. **Install Jupyter (dev dependency)**

   ```bash
   poetry add --group dev notebook ipykernel
   ```

2. **Register the Poetry environment as a Jupyter kernel**

   ```bash
   poetry run python -m ipykernel install --user --name "lsjssp-poetry" --display-name "Python (LSJSSP Poetry)"
   ```

3. **Launch Jupyter**

   ```bash
   poetry run jupyter notebook
   ```

4. **Select the correct kernel**
   In the notebook interface, go to **Kernel → Change Kernel → Python (LSJSSP Poetry)**.
   This ensures your notebooks use the exact same dependencies as the rest of the project.

## 📂 Repository structure
```
config/                # YAML configuration files
src/
  genetic/             # Genetic algorithm components
  jobshop/             # Job shop parameter generation
  scheduling/          # Chromosome decoding and scheduling logic
  plotting/            # Gantt and fitness evolution plotting
main.py                # Example script tying everything together
tests/                 # Pytest-based test suite
```

## Usage
1. Adjust `config/jobshop/js_params_01.yaml` and `config/genetic_algorithm/ga_config_01.yaml` as needed.
2. Run the demonstration script:
   ```bash
   python main.py
   ```
   The script prints the best makespan found and, if enabled, penalty values. An interactive Gantt chart (Plotly HTML) and fitness evolution plot are saved to `results/schedule/` and `results/fitness_evolution/` respectively.

## 🧩 How it works (high-level)

- **Chromosome structure** – Encodes two components:

  * **LHS (Left-Hand Side)**: Continuous values representing lot sizes.
  * **RHS (Right-Hand Side)**: Discrete sequence of operations for all jobs.
- **Scheduler / Decoder** – Transforms a chromosome into a feasible *semi-active* schedule that respects job precedence, sequence-dependent setup times, and optional shift constraints.
- **Genetic operators** – Specialized crossover and mutation methods that modify LHS and/or RHS while maintaining feasibility.
- **Fitness evaluation** – Primarily minimizes makespan, with optional penalty terms for violations related to shift constraints.

## 📚 References
Fantahun M. Defersha & Mingyuan Chen (2012) *Jobshop lot streaming with routing flexibility, sequence-dependent setups, machine release dates and lag time*, International Journal of Production Research, 50:8, 2331-2352.


