import numpy as np
import random


# ============================================================
# Invasive Weed Optimization (IWO) for Subset Sum Problem
# University-Style Version with Slide-Like Tables
# ============================================================


# ============================================================
# Problem Definition
# ============================================================

numbers = [3, 34, 4, 12, 5, 2]
target = 10
dimension = len(numbers)


# ============================================================
# IWO Parameters
# ============================================================

T = 10                  # Total number of iterations
np_init = 4             # Initial population size
P_max = 6               # Maximum population size
S_min = 1               # Minimum number of seeds
S_max = 3               # Maximum number of seeds
n = 3                   # Nonlinear modulation index
sigma_initial = 1.0     # Initial standard deviation
sigma_final = 0.05      # Final standard deviation


# ============================================================
# Fitness Function
# ------------------------------------------------------------
# A solution is a binary vector:
# 1 -> number is selected
# 0 -> number is not selected
#
# selected_sum = sum of chosen numbers
# difference   = |target - selected_sum|
#
# Since IWO in the slides prefers larger fitness values,
# we define:
# fitness = 1 / (1 + difference)
#
# Exact solution => difference = 0 => fitness = 1.0
# ============================================================

def calculate_fitness(solution):
    selected_sum = sum(numbers[i] for i in range(dimension) if solution[i] == 1)
    difference = abs(target - selected_sum)
    fitness = 1.0 / (1.0 + difference)
    return fitness, selected_sum, difference


# ============================================================
# Initialize Population
# ============================================================

def initialize_population(pop_size, dim):
    population = []
    for _ in range(pop_size):
        weed = np.random.randint(0, 2, size=dim)
        population.append(weed)
    return population


# ============================================================
# Calculate Sigma_t (same nonlinear formula as slides)
# ============================================================

def calculate_sigma(t, T, n, sigma_initial, sigma_final):
    sigma_t = (((T - t) / T) ** n) * (sigma_initial - sigma_final) + sigma_final
    return sigma_t


# ============================================================
# Calculate Seed Count
# ------------------------------------------------------------
# Better fitness -> more seeds
# Worse fitness  -> fewer seeds
# ============================================================

def calculate_seed_count(fitness, best_fitness, worst_fitness, S_min, S_max):
    if best_fitness == worst_fitness:
        return S_min

    seed_value = S_min + (fitness - worst_fitness) * (S_max - S_min) / (best_fitness - worst_fitness)
    return int(np.floor(seed_value))


# ============================================================
# Convert Continuous Vector Back to Binary
# ============================================================

def binary_threshold(vector):
    return np.where(vector >= 0.5, 1, 0).astype(int)


# ============================================================
# Generate One Child from One Parent
# ============================================================

def generate_child(parent, sigma):
    noise = np.random.normal(loc=0.0, scale=sigma, size=dimension)
    child_continuous = parent.astype(float) + noise
    child_binary = binary_threshold(child_continuous)
    return child_binary, noise


# ============================================================
# Convert Binary Vector to Selected Subset
# ============================================================

def extract_subset(solution):
    return [numbers[i] for i in range(dimension) if solution[i] == 1]


# ============================================================
# Evaluate Population
# Each row contains:
# (label, solution, fitness, selected_sum, difference)
# ============================================================

def evaluate_population(population, labels=None):
    evaluated = []

    for idx, weed in enumerate(population):
        fitness, selected_sum, difference = calculate_fitness(weed)

        if labels is None:
            label = f"A{idx+1}"
        else:
            label = labels[idx]

        evaluated.append({
            "label": label,
            "solution": weed,
            "fitness": fitness,
            "sum": selected_sum,
            "difference": difference,
            "subset": extract_subset(weed)
        })

    return evaluated


# ============================================================
# Print Population in Table Format
# Similar to slide-style A1, A2, A3, ...
# ============================================================

def print_population_table(evaluated_population, title="Population Table"):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(f"{'Weed':<6} {'Binary Solution':<22} {'Selected Subset':<25} {'Sum':<8} {'Difference':<12} {'Fitness':<10}")
    print("-" * 110)

    for item in evaluated_population:
        label = item["label"]
        solution = str(item["solution"].tolist())
        subset = str(item["subset"])
        selected_sum = item["sum"]
        difference = item["difference"]
        fitness = item["fitness"]

        print(f"{label:<6} {solution:<22} {subset:<25} {selected_sum:<8} {difference:<12} {fitness:<10.4f}")

    print("=" * 110)


# ============================================================
# Print Seed Counts
# ============================================================

def print_seed_table(seed_info, title="Seed Production Table"):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"{'Weed':<6} {'Fitness':<12} {'Number of Seeds':<18}")
    print("-" * 90)

    for row in seed_info:
        print(f"{row['label']:<6} {row['fitness']:<12.4f} {row['seed_count']:<18}")

    print("=" * 90)


# ============================================================
# Sort Population by Fitness Descending
# ============================================================

def sort_by_fitness(evaluated_population):
    return sorted(evaluated_population, key=lambda x: x["fitness"], reverse=True)


# ============================================================
# Competitive Exclusion
# Keep only the best P_max weeds
# ============================================================

def competitive_exclusion(evaluated_population, P_max):
    sorted_population = sort_by_fitness(evaluated_population)
    return sorted_population[:P_max]


# ============================================================
# Main IWO Procedure
# ============================================================

def iwo_subset_sum_university():

    # -----------------------------
    # Step 1: Initial population
    # -----------------------------
    population = initialize_population(np_init, dimension)

    # Global best solution tracker
    global_best = None

    # Evaluate initial population
    evaluated_population = evaluate_population(population)
    evaluated_population = sort_by_fitness(evaluated_population)

    print_population_table(evaluated_population, title="Initial Population")

    # Update global best
    global_best = evaluated_population[0]

    # -----------------------------
    # Main iterative process
    # -----------------------------
    for t in range(1, T + 1):
        print("\n\n" + "#" * 120)
        print(f"ITERATION {t}")
        print("#" * 120)

        # Step 2: Compute sigma_t
        sigma_t = calculate_sigma(t, T, n, sigma_initial, sigma_final)
        print(f"\nSigma_{t} = {sigma_t:.6f}")

        # Step 3: Evaluate and rank current population
        evaluated_population = sort_by_fitness(evaluated_population)

        print_population_table(
            evaluated_population,
            title=f"Current Population Before Reproduction - Iteration {t}"
        )

        best_fitness = evaluated_population[0]["fitness"]
        worst_fitness = evaluated_population[-1]["fitness"]

        # Step 4: Determine seed counts
        seed_info = []
        all_children = []
        child_counter = 1

        for item in evaluated_population:
            seed_count = calculate_seed_count(
                fitness=item["fitness"],
                best_fitness=best_fitness,
                worst_fitness=worst_fitness,
                S_min=S_min,
                S_max=S_max
            )

            seed_info.append({
                "label": item["label"],
                "fitness": item["fitness"],
                "seed_count": seed_count
            })

            # Step 5: Generate children
            for _ in range(seed_count):
                child_solution, noise = generate_child(item["solution"], sigma_t)

                fitness, selected_sum, difference = calculate_fitness(child_solution)

                all_children.append({
                    "label": f"C{child_counter}",
                    "solution": child_solution,
                    "fitness": fitness,
                    "sum": selected_sum,
                    "difference": difference,
                    "subset": extract_subset(child_solution),
                    "parent": item["label"],
                    "noise": noise
                })

                child_counter += 1

        print_seed_table(seed_info, title=f"Seed Production - Iteration {t}")

        # Print offspring table
        print("\n" + "=" * 120)
        print(f"Offspring Population - Iteration {t}")
        print("=" * 120)
        print(f"{'Child':<8} {'Parent':<8} {'Binary Solution':<22} {'Selected Subset':<25} {'Sum':<8} {'Diff':<8} {'Fitness':<10}")
        print("-" * 120)

        for child in all_children:
            print(f"{child['label']:<8} {child['parent']:<8} {str(child['solution'].tolist()):<22} "
                  f"{str(child['subset']):<25} {child['sum']:<8} {child['difference']:<8} {child['fitness']:<10.4f}")

        print("=" * 120)

        # Step 6: Combine parents and offspring
        combined_population = []

        for item in evaluated_population:
            combined_population.append(item)

        for child in all_children:
            combined_population.append(child)

        print_population_table(
            combined_population,
            title=f"Combined Population (Parents + Offspring) - Iteration {t}"
        )

        # Step 7: Sort combined population by fitness
        sorted_combined = sort_by_fitness(combined_population)

        print_population_table(
            sorted_combined,
            title=f"Sorted Combined Population by Fitness - Iteration {t}"
        )

        # Step 8: Competitive exclusion
        next_generation = competitive_exclusion(sorted_combined, P_max)

        # Re-label survivors as A1, A2, A3, ...
        relabeled_population = []
        for idx, item in enumerate(next_generation):
            relabeled_population.append({
                "label": f"A{idx+1}",
                "solution": item["solution"],
                "fitness": item["fitness"],
                "sum": item["sum"],
                "difference": item["difference"],
                "subset": item["subset"]
            })

        evaluated_population = relabeled_population

        print_population_table(
            evaluated_population,
            title=f"Population After Competitive Exclusion - Iteration {t}"
        )

        # Step 9: Update global best
        current_best = evaluated_population[0]
        if current_best["fitness"] > global_best["fitness"]:
            global_best = current_best

        # Step 10: Check stopping condition
        if global_best["difference"] == 0:
            print("\nExact solution reached. Stopping condition satisfied.")
            break

    # -----------------------------
    # Final result
    # -----------------------------
    print("\n\n" + "#" * 120)
    print("FINAL BEST SOLUTION")
    print("#" * 120)

    print(f"Weed Label       : {global_best['label']}")
    print(f"Binary Solution  : {global_best['solution'].tolist()}")
    print(f"Selected Subset  : {global_best['subset']}")
    print(f"Subset Sum       : {global_best['sum']}")
    print(f"Target           : {target}")
    print(f"Difference       : {global_best['difference']}")
    print(f"Fitness          : {global_best['fitness']:.4f}")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    iwo_subset_sum_university()
