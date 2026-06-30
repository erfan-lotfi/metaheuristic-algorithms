# Metaheuristic Algorithms Collection

A comprehensive collection of implementations of metaheuristic algorithms with an educational, step‑by‑step approach. This repository contains 10 well‑known optimization algorithms, each designed to solve a specific problem. The code is written to display intermediate outputs, making it ideal for understanding the internal processes of each algorithm.

---

## 📚 List of Algorithms

| Algorithm | File | Problem | Description |
|-----------|------|---------|-------------|
| **ABC** (Artificial Bee Colony) | `ABC-grid-obstacle-avoidance.py` | Grid pathfinding with obstacles | Finds an optimal path from start to goal with minimum moves and obstacle collisions using artificial bee colony. |
| **COA** (Cuckoo Optimization Algorithm) | `COA-Circle-Packing-in-a-Square.py` | Circle packing in a square | Finds the maximum possible radius for packing N circles inside a unit square without overlap. |
| **GA** (Genetic Algorithm) | `GA-angry-birds.py` | Angry Birds projectile simulation | Tunes launch angles and velocities to hit pigs with minimal shots using genetic algorithm. |
| **GOA** (Grasshopper Optimization Algorithm) | `GOA-sudoko.py` | Sudoku puzzle solving | Completes a Sudoku grid with minimum violations of row, column, and sub‑grid constraints. |
| **GWO** (Grey Wolf Optimizer) | `GWO-maximum-clique.py` | Maximum clique problem | Finds the largest set of vertices that form a complete subgraph in a graph. |
| **IWO** (Invasive Weed Optimization) | `IWO-subset-sum.py` | Subset sum problem | Finds a subset of numbers whose sum equals a target value. |
| **MFO** (Moth‑Flame Optimization) | `moth-flame-optimization-maximum-cut.py` | Maximum cut problem | Finds a cut with maximum total weight in a graph. |
| **PSO** (Particle Swarm Optimization) | `pso-job-scheduling.py` | Flow shop scheduling | Finds the optimal job sequence to minimise makespan in a flow shop environment. |
| **SA** (Simulated Annealing) | `SA-Supply-Chain-Stabilization.py` | Supply chain stabilisation | Optimises order quantities in a multi‑echelon supply chain to reduce holding, shortage, and fluctuation costs. |
| **SFLA** (Shuffled Frog Leaping Algorithm) | `SFLA-super-mario.py` | Super Mario pathfinding | Finds an optimal path through a grid collecting coins and avoiding enemies. |

---

## 🧠 Key Features

- **Step‑by‑step output** – All algorithms display detailed intermediate tables and logs, which are extremely helpful for learning and analysing algorithm behaviour.
- **Pure Python implementation** – No heavy dependencies like `scikit‑learn` or `tensorflow`; only standard libraries (`random`, `math`, `copy`) are used.
- **Diverse problem types** – From continuous and discrete optimisation to combinatorial and graph problems.
- **Reproducible randomness** – Every algorithm includes a `seed` parameter to ensure repeatable results.
- **Visual progress tracking** – Population tables, convergence matrices, and best‑solution summaries are printed at each iteration.

---

## 📦 Prerequisites

- Python 3.6 or higher
- Standard libraries (all included by default):
  - `random`
  - `math`
  - `copy`
  - `typing` (used in MFO)

No additional packages are required. Simply run any file in a terminal or Jupyter environment.

---

## 🚀 How to Run

Each file is self‑contained and includes a `if __name__ == "__main__"` block with a default example. To run any algorithm, execute the corresponding file in your terminal:

```bash
python ABC-grid-obstacle-avoidance.py
python COA-Circle-Packing-in-a-Square.py
python GA-angry-birds.py
# and so on for the other files
```

### Quick Example for PSO

```python
from pso-job-scheduling import pso_flow_shop_scheduling

processing_times = [
    [6, 3, 2],
    [2, 4, 1],
    [8, 2, 5]
]

best_seq, best_cost = pso_flow_shop_scheduling(
    processing_times=processing_times,
    num_particles=5,
    max_iter=20,
    seed=42
)

print(f"Best sequence: {best_seq}, Makespan: {best_cost}")
```

---

## 📁 File Structure

```
.
├── ABC-grid-obstacle-avoidance.py          # Artificial Bee Colony
├── COA-Circle-Packing-in-a-Square.py       # Cuckoo Optimization Algorithm
├── GA-angry-birds.py                       # Genetic Algorithm
├── GOA-sudoko.py                           # Grasshopper Optimization Algorithm
├── GWO-maximum-clique.py                   # Grey Wolf Optimizer
├── IWO-subset-sum.py                       # Invasive Weed Optimization
├── moth-flame-optimization-maximum-cut.py  # Moth‑Flame Optimization
├── pso-job-scheduling.py                   # Particle Swarm Optimization
├── SA-Supply-Chain-Stabilization.py        # Simulated Annealing
├── SFLA-super-mario.py                     # Shuffled Frog Leaping Algorithm
└── README.md                               # This file
```

---

## 📊 Sample Output

Each algorithm produces detailed intermediate output. Here is an excerpt from the ABC algorithm:

```
============================================================
STEP 1 & 2: INITIALIZATION OF COLONY AND FOOD SOURCES
============================================================
Grid size          : 5 x 5
Start              : (0, 0)
Goal               : (4, 4)
Path length (D)    : 10
...
Initial Food Sources
----------------------------------------------------------------------
Src  Moves                F(X)      fit(X)      Trial
----------------------------------------------------------------------
1    UURDLURRLD           23         0.041667    0
2    RLDRURDLLU           18         0.052632    0
...
```

At the end of each run, the **best final solution** is printed with full details, including path trace, valid steps, and goal achievement status.

---

## 🧪 Parameter Tuning

All algorithms expose tunable parameters. Refer to the class docstring or the `__init__` method of each algorithm for details. Some examples:

- **ABC**: `colony_size`, `employed_bees`, `onlooker_bees`, `max_cycles`, `limit`
- **GA**: `pop_size`, `crossover_rate`, `mutation_rate`, `generations`
- **PSO**: `num_particles`, `max_iter`, `w`, `c1`, `c2`
- **SA**: `INITIAL_TEMPERATURE`, `COOLING_STEP`, `ITERATIONS_PER_TEMPERATURE`

---

## 📖 Study Guide

If you want to gain a deeper understanding of each algorithm, we recommend:

1. Read the problem description first.
2. Run the code and observe the step‑by‑step output.
3. Compare each step with the theoretical formulation of the algorithm.
4. Modify parameters and observe their effect on convergence behaviour.

---

**🌟 If you find this repository useful, please give it a ⭐ so that others can benefit from it too.**
