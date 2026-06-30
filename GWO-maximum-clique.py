import random


def format_vector(vec, precision=3):
    """
    Format a vector for nicer slide-style printing.
    """
    formatted = []
    for x in vec:
        if isinstance(x, float):
            formatted.append(round(x, precision))
        else:
            formatted.append(x)
    return formatted


def is_clique(vertices, graph):
    """
    Check whether a set of vertices forms a clique.
    """
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if graph[vertices[i]][vertices[j]] == 0:
                return False
    return True


def repair_to_clique(solution, graph, verbose=False):
    """
    Repair a binary solution so that selected vertices form a valid clique.
    """
    n = len(solution)
    selected = [i for i, bit in enumerate(solution) if bit == 1]

    if verbose:
        print("    Selected vertices before repair:", selected)

    # Remove conflicting vertices until clique property holds
    while not is_clique(selected, graph):
        conflict_count = {v: 0 for v in selected}

        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                u = selected[i]
                v = selected[j]
                if graph[u][v] == 0:
                    conflict_count[u] += 1
                    conflict_count[v] += 1

        worst_vertex = max(conflict_count, key=conflict_count.get)

        if verbose:
            print("    Conflict count:", conflict_count)
            print("    Removed vertex:", worst_vertex)

        selected.remove(worst_vertex)

    # Greedily add compatible vertices
    remaining = [v for v in range(n) if v not in selected]
    improved = True

    while improved:
        improved = False
        for v in remaining[:]:
            if all(graph[u][v] == 1 for u in selected):
                selected.append(v)
                remaining.remove(v)
                improved = True
                if verbose:
                    print("    Added compatible vertex:", v)

    repaired = [0] * n
    for v in selected:
        repaired[v] = 1

    if verbose:
        print("    Selected vertices after repair:", selected)
        print("    Repaired binary solution:", repaired)

    return repaired


def fitness(solution, graph):
    """
    Fitness = size of repaired clique.
    """
    repaired = repair_to_clique(solution, graph, verbose=False)
    return sum(repaired), repaired


def continuous_to_binary(position):
    """
    Convert continuous position to binary using threshold 0.5.
    """
    return [1 if x >= 0.5 else 0 for x in position]


def initialize_population(num_wolves, dimension):
    """
    Initialize population as binary vectors.
    """
    return [[random.randint(0, 1) for _ in range(dimension)] for _ in range(num_wolves)]


def print_population(population, title):
    """
    Print population in a clean format.
    """
    print(f"\n{title}")
    print("-" * len(title))
    for i, wolf in enumerate(population):
        print(f"Wolf {i + 1}: {wolf}")


def gwo_maximum_clique_slide_style(graph, num_wolves=4, max_iter=3, seed=42):
    """
    Grey Wolf Optimizer for Maximum Clique with slide-style vector logs.
    """
    random.seed(seed)
    n = len(graph)

    # Step 1: Initial population
    population = initialize_population(num_wolves, n)
    print_population(population, "Initial Population")

    global_best_solution = None
    global_best_score = -1

    for iteration in range(max_iter):
        print("\n" + "=" * 100)
        print(f"ITERATION {iteration + 1}/{max_iter}")
        print("=" * 100)

        # Step 2: Evaluate fitness and identify alpha, beta, delta
        alpha_pos, beta_pos, delta_pos = None, None, None
        alpha_score, beta_score, delta_score = -1, -1, -1

        print("\nFitness Evaluation")
        print("------------------")

        repaired_population = []

        for i, wolf in enumerate(population):
            score, repaired = fitness(wolf, graph)
            repaired_population.append(repaired)

            print(f"Wolf {i + 1}:")
            print(f"  Original solution = {wolf}")
            print(f"  Repaired solution = {repaired}")
            print(f"  Fitness           = {score}")

            if score > alpha_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = alpha_score, alpha_pos
                alpha_score, alpha_pos = score, repaired[:]
            elif score > beta_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = score, repaired[:]
            elif score > delta_score:
                delta_score, delta_pos = score, repaired[:]

        print("\nLeaders")
        print("-------")
        print(f"Alpha = {alpha_pos}, fitness = {alpha_score}")
        print(f"Beta  = {beta_pos}, fitness = {beta_score}")
        print(f"Delta = {delta_pos}, fitness = {delta_score}")

        if alpha_score > global_best_score:
            global_best_score = alpha_score
            global_best_solution = alpha_pos[:]

        # Step 3: Update parameter a
        a = 2 - iteration * (2 / max_iter)
        print("\nParameter Update")
        print("----------------")
        print(f"a = {round(a, 3)}")

        # Step 4: Update all wolves
        new_population = []

        print("\nPosition Update")
        print("---------------")

        for i in range(num_wolves):
            wolf = population[i]

            # Generate full vectors A and C for alpha, beta, delta
            A1 = [2 * a * random.random() - a for _ in range(n)]
            C1 = [2 * random.random() for _ in range(n)]

            A2 = [2 * a * random.random() - a for _ in range(n)]
            C2 = [2 * random.random() for _ in range(n)]

            A3 = [2 * a * random.random() - a for _ in range(n)]
            C3 = [2 * random.random() for _ in range(n)]

            # Vector calculations exactly in slide style
            D_alpha = [abs(C1[j] * alpha_pos[j] - wolf[j]) for j in range(n)]
            D_beta = [abs(C2[j] * beta_pos[j] - wolf[j]) for j in range(n)]
            D_delta = [abs(C3[j] * delta_pos[j] - wolf[j]) for j in range(n)]

            X1 = [alpha_pos[j] - A1[j] * D_alpha[j] for j in range(n)]
            X2 = [beta_pos[j] - A2[j] * D_beta[j] for j in range(n)]
            X3 = [delta_pos[j] - A3[j] * D_delta[j] for j in range(n)]

            X_new = [(X1[j] + X2[j] + X3[j]) / 3.0 for j in range(n)]

            binary_solution = continuous_to_binary(X_new)
            repaired_solution = repair_to_clique(binary_solution, graph, verbose=False)
            score, repaired_solution = fitness(repaired_solution, graph)

            print(f"\nWolf {i + 1}")
            print(f"Current Position = {wolf}")
            print(f"A1      = {format_vector(A1)}")
            print(f"C1      = {format_vector(C1)}")
            print(f"A2      = {format_vector(A2)}")
            print(f"C2      = {format_vector(C2)}")
            print(f"A3      = {format_vector(A3)}")
            print(f"C3      = {format_vector(C3)}")
            print(f"D_alpha = {format_vector(D_alpha)}")
            print(f"D_beta  = {format_vector(D_beta)}")
            print(f"D_delta = {format_vector(D_delta)}")
            print(f"X1      = {format_vector(X1)}")
            print(f"X2      = {format_vector(X2)}")
            print(f"X3      = {format_vector(X3)}")
            print(f"X(t+1)  = {format_vector(X_new)}")
            print(f"Binary  = {binary_solution}")
            print(f"Repair  = {repaired_solution}")
            print(f"Fitness = {score}")

            new_population.append(repaired_solution)

        population = new_population
        print_population(population, f"Population After Iteration {iteration + 1}")

        print("\nIteration Summary")
        print("-----------------")
        print(f"Best fitness in iteration = {alpha_score}")
        print(f"Best solution in iteration = {alpha_pos}")

    clique_vertices = [i for i, bit in enumerate(global_best_solution) if bit == 1]

    print("\n" + "=" * 100)
    print("FINAL RESULT")
    print("=" * 100)
    print(f"Best binary solution = {global_best_solution}")
    print(f"Best clique vertices = {clique_vertices}")
    print(f"Best clique size     = {global_best_score}")

    return global_best_solution, clique_vertices, global_best_score


# Example usage
if __name__ == "__main__":
    graph = [
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ]

    gwo_maximum_clique_slide_style(
        graph=graph,
        num_wolves=4,
        max_iter=3,
        seed=42
    )
