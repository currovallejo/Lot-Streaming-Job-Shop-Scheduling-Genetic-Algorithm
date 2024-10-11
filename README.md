# GENETIC ALGORITHM for LOT STREAMING in JOB SHOP SCHEDULING 
Set of modules for solving the lot streaming job shop scheduling problem through a genetic algorithm.

## Features
- lot streaming (setting max number of lots for each job)
- Possibility of handling shift constraints
- Possibility of handling sequence dependent setup times
- Python code PEP8 compliant

| ![newplot](https://github.com/user-attachments/assets/8f404e91-f633-455d-ae4d-4373c8421596) | 
|:--:| 
| *LSJSP Scheduling with shift constraints and sequence dependent setup times* |

## Architecture
- params.py | generation of random job shop parameters
- chromosome_generator.py | generation of random chromosomes for GA population
- decoder.py | decodification and evaluation of each chromosome (solution) through active scheduling algorithm (no unnecessary idle times)
- genetic_operators.py | cross-over and mutation operators
- genetic_algorithm.py | genetic algorithm and main script
- plot.py | plotting of Gantt chart with plotly express timeline
![alt text](image.png)
|:--:| 
| *Architecture of the Program. main.py not implemented, used instead main() in genetic_algorithm.py* |

## JOB SHOP PAREMETERS
**Inputs**
- number of machines
- number of jobs
- max number of lots for each job (same for all jobs)
- seed (partial control of random generation)

**Outputs**
- processing times
- setup times
- sequences
  
*Demand is set to 50 units for each job and can be changed after generating the parameters object*

# TRY IT YOURSELF!
Take a look at the GA_guide.ipynb jupyter notebook. There you will find the whole program explained step by step.

## References
Fantahun M. Defersha & Mingyuan Chen (2012) Jobshop lot streaming with routing flexibility, sequence-dependent setups, machine release dates and lag time, International Journal of Production Research, 50:8, 2331-2352

Params.py is a file originally created by Bruno Scalia Carneiro Ferreira Leite for his [jobshop package](https://github.com/bruscalia/jobshop)

