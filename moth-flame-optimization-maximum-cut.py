import math
import random
from typing import List, Tuple, Optional


class MFOMaxCut:
    """
    Moth-Flame Optimization for the Maximum Cut problem.

    Each moth represents a candidate partition of graph vertices.
    For a graph with n vertices, a solution is a binary vector of length n.

    Example:
        solution = [0, 1, 0, 1, 1]

    Meaning:
        Vertices with value 0 are in set A.
        Vertices with value 1 are in set B.

    The fitness value is the total weight of edges crossing between A and B.
    """

    def __init__(
        self,
        graph: List[List[float]],
        n_moths: int = 10,
        max_iter: int = 100,
        b: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the MFO solver.

        Parameters:
            graph:
                Adjacency matrix of the graph.
                graph[i][j] represents the weight of edge (i, j).
                If graph[i][j] = 0, there is no edge.

            n_moths:
                Number of moths in the population.

            max_iter:
                Maximum number of iterations.

            b:
                Spiral constant in the MFO formula.
                In the slides, b = 1.

            seed:
                Random seed for reproducibility.

            verbose:
                If True, prints iteration information.
        """

        if seed is not None:
            random.seed(seed)

        self.graph = graph
        self.n_vertices = len(graph)
        self.n_moths = n_moths
        self.max_iter = max_iter
        self.b = b
        self.verbose = verbose

        self._validate_graph()

    def _validate_graph(self):
        """
        Validate that the input graph is a square adjacency matrix.
        """

        if self.n_vertices == 0:
            raise ValueError("Graph cannot be empty.")

        for row in self.graph:
            if len(row) != self.n_vertices:
                raise ValueError("Graph must be a square adjacency matrix.")

    # ------------------------------------------------------------
    # Step 1: Initialize moth population
    # ------------------------------------------------------------
    def initialize_population(self) -> List[List[int]]:
        """
        Create the initial population of moths.

        Since Max-Cut is a binary problem, each moth is a binary vector.
        Each element is either 0 or 1.
        """

        population = []

        for _ in range(self.n_moths):
            moth = [random.randint(0, 1) for _ in range(self.n_vertices)]
            population.append(moth)

        return population

    # ------------------------------------------------------------
    # Step 2: Fitness function for Maximum Cut
    # ------------------------------------------------------------
    def fitness(self, solution: List[int]) -> float:
        """
        Calculate the cut value of a binary solution.

        An edge contributes to the cut if its two endpoints are assigned
        to different partitions.

        For each edge (i, j):
            if solution[i] != solution[j], then edge (i, j) is cut.

        Returns:
            Total cut weight.
        """

        cut_value = 0.0

        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if solution[i] != solution[j]:
                    cut_value += self.graph[i][j]

        return cut_value

    # ------------------------------------------------------------
    # Step 3: Sort moths based on fitness
    # ------------------------------------------------------------
    def sort_population(
        self,
        population: List[List[int]]
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Sort the population in descending order of fitness.

        Since Max-Cut is a maximization problem, higher fitness is better.
        """

        scored_population = []

        for moth in population:
            score = self.fitness(moth)
            scored_population.append((moth, score))

        scored_population.sort(key=lambda item: item[1], reverse=True)

        sorted_population = [item[0] for item in scored_population]
        sorted_fitness = [item[1] for item in scored_population]

        return sorted_population, sorted_fitness

    # ------------------------------------------------------------
    # Step 4: Convert continuous value to binary value
    # ------------------------------------------------------------
    def continuous_to_binary(self, value: float) -> int:
        """
        Convert a continuous MFO output into a binary value.

        In the 8-queen slides, continuous values were rounded and mapped
        into valid queen positions.

        For Max-Cut, the valid search space is binary:
            0 or 1

        Here we use thresholding:
            if value >= 0.5 -> 1
            otherwise      -> 0

        Before thresholding, the value is clipped to [0, 1].
        """

        value = max(0.0, min(1.0, value))

        if value >= 0.5:
            return 1
        else:
            return 0

    # ------------------------------------------------------------
    # Step 5: MFO position update equation
    # ------------------------------------------------------------
    def update_moth_position(
        self,
        moth: List[int],
        flame: List[int],
        r: float,
        debug: bool = False
    ) -> List[int]:
        """
        Update a moth position using the MFO spiral equation.

        The equation from the slides is:

            M_i,new = S(M_i, F_j)
                    = D_i * e^(b*t) * cos(2*pi*t) + F_j

        where:

            D_i = |F_j - M_i|

            t = random[r, 1]

            b = 1

        For each dimension d:

            D_d = |flame[d] - moth[d]|

            x_new = D_d * exp(b * t) * cos(2*pi*t) + flame[d]

        Since Max-Cut is binary, x_new is mapped to 0 or 1.
        """

        new_moth = []

        for d in range(self.n_vertices):
            M_id = moth[d]
            F_jd = flame[d]

            # Distance between moth and flame in this dimension
            D_id = abs(F_jd - M_id)

            # Random parameter t in the interval [r, 1]
            t = random.uniform(r, 1)

            # MFO spiral update equation
            x_new = D_id * math.exp(self.b * t) * math.cos(2 * math.pi * t) + F_jd

            # Convert continuous value to binary
            bit = self.continuous_to_binary(x_new)

            if debug:
                print(
                    f"    dim={d + 1:2d} | "
                    f"M={M_id} | F={F_jd} | D=|{F_jd}-{M_id}|={D_id} | "
                    f"t={t:.4f} | x_new={x_new:.4f} | bit={bit}"
                )

            new_moth.append(bit)

        return new_moth

    # ------------------------------------------------------------
    # Step 6: Calculate number of flames
    # ------------------------------------------------------------
    def calculate_flame_number(self, iteration: int) -> int:
        """
        Calculate the number of active flames.

        Formula used in the slides:

            flame_no = round(
                N - iteration * ((N - 1) / max_iter)
            )

        where:
            N = number of moths

        As iterations increase, flame_no decreases.
        This gradually shifts the algorithm from exploration to exploitation.
        """

        flame_no = round(
            self.n_moths - iteration * ((self.n_moths - 1) / self.max_iter)
        )

        flame_no = max(1, flame_no)

        return flame_no

    # ------------------------------------------------------------
    # Step 7: Calculate r parameter
    # ------------------------------------------------------------
    def calculate_r(self, iteration: int) -> float:
        """
        Calculate the r parameter.

        According to the slide-style formula:

            r = -1 - iteration / max_iter

        At early iterations:
            r is close to -1.

        At final iterations:
            r approaches -2.
        """

        r = -1 - (iteration / self.max_iter)
        return r

    # ------------------------------------------------------------
    # Step 8: Run the MFO algorithm
    # ------------------------------------------------------------
    def run(self) -> Tuple[List[int], float, List[float]]:
        """
        Execute the MFO algorithm.

        Returns:
            best_solution:
                Best binary partition found.

            best_fitness:
                Best cut value found.

            history:
                Best fitness value at each iteration.
        """

        # Initialize moths randomly
        moths = self.initialize_population()

        # Sort initial moths
        moths, moth_fitness = self.sort_population(moths)

        # Initial flames are the sorted moths
        flames = [moth[:] for moth in moths]
        flame_fitness = moth_fitness[:]

        # Store the global best solution
        best_solution = flames[0][:]
        best_fitness = flame_fitness[0]

        # Store convergence history
        history = [best_fitness]

        if self.verbose:
            print("Initial population:")
            for i, moth in enumerate(moths):
                print(f"  Moth {i + 1}: {moth} | fitness = {moth_fitness[i]}")
            print("-" * 80)

        # Main loop
        for iteration in range(1, self.max_iter + 1):

            # Calculate r parameter
            r = self.calculate_r(iteration)

            # Calculate number of flames
            flame_no = self.calculate_flame_number(iteration)

            if self.verbose:
                print(f"Iteration {iteration}")
                print(f"  r = {r:.4f}")
                print(f"  flame_no = {flame_no}")

            new_moths = []

            # Update each moth position
            for i in range(self.n_moths):

                # In standard MFO:
                # If moth index is less than flame_no, it follows the corresponding flame.
                # Otherwise, it follows the last active flame.
                if i < flame_no:
                    flame_index = i
                else:
                    flame_index = flame_no - 1

                moth = moths[i]
                flame = flames[flame_index]

                if self.verbose:
                    print(f"  Updating Moth {i + 1} using Flame {flame_index + 1}")
                    print(f"    Moth  = {moth}")
                    print(f"    Flame = {flame}")
                    print(f"    Flame fitness = {flame_fitness[flame_index]}")

                new_moth = self.update_moth_position(
                    moth=moth,
                    flame=flame,
                    r=r,
                    debug=self.verbose
                )

                new_moths.append(new_moth)

                if self.verbose:
                    print(f"    New moth = {new_moth}")
                    print(f"    New fitness = {self.fitness(new_moth)}")

            # Sort new moths
            new_moths_sorted, new_moths_fitness = self.sort_population(new_moths)

            # Combine old flames and new moths
            combined_population = flames + new_moths_sorted

            # Sort combined population
            combined_sorted, combined_fitness = self.sort_population(combined_population)

            # Select the best n_moths solutions as new flames
            flames = [solution[:] for solution in combined_sorted[:self.n_moths]]
            flame_fitness = combined_fitness[:self.n_moths]

            # Moths for next iteration are the newly generated moths
            moths = [solution[:] for solution in new_moths_sorted]
            moth_fitness = new_moths_fitness[:]

            # Update global best
            if flame_fitness[0] > best_fitness:
                best_solution = flames[0][:]
                best_fitness = flame_fitness[0]

            history.append(best_fitness)

            if self.verbose:
                print("  Best flame after sorting:")
                print(f"    Best solution = {best_solution}")
                print(f"    Best fitness  = {best_fitness}")
                print("-" * 80)

        return best_solution, best_fitness, history

    # ------------------------------------------------------------
    # Utility: Decode binary solution into two vertex sets
    # ------------------------------------------------------------
    def decode_solution(self, solution: List[int]) -> Tuple[List[int], List[int]]:
        """
        Convert a binary solution into two sets of vertices.

        Vertices with value 0 are placed in set A.
        Vertices with value 1 are placed in set B.

        Vertex indices are returned as 0-based indices.
        """

        set_a = []
        set_b = []

        for i, value in enumerate(solution):
            if value == 0:
                set_a.append(i)
            else:
                set_b.append(i)

        return set_a, set_b


# --------------------------------------------------------------------
# Helper function for building graph from edge list
# --------------------------------------------------------------------
def build_adjacency_matrix(
    n_vertices: int,
    edges: List[Tuple[int, int, float]],
    one_based: bool = False
) -> List[List[float]]:
    """
    Build an adjacency matrix from an edge list.

    Parameters:
        n_vertices:
            Number of vertices.

        edges:
            List of edges in the form:
                (u, v, weight)

        one_based:
            If True, vertices in the edge list are assumed to start from 1.
            If False, vertices are assumed to start from 0.

    Returns:
        Adjacency matrix.
    """

    graph = [[0.0 for _ in range(n_vertices)] for _ in range(n_vertices)]

    for u, v, w in edges:
        if one_based:
            u -= 1
            v -= 1

        graph[u][v] = w
        graph[v][u] = w

    return graph


# --------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------
if __name__ == "__main__":

    # Example 1:
    # An unweighted graph represented as an adjacency matrix.
    graph = [
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ]

    solver = MFOMaxCut(
        graph=graph,
        n_moths=7,
        max_iter=50,
        b=1.0,
        seed=42,
        verbose=True
    )

    best_solution, best_fitness, history = solver.run()

    set_a, set_b = solver.decode_solution(best_solution)

    print("\nFinal Result")
    print("=" * 80)
    print(f"Best solution : {best_solution}")
    print(f"Best cut value: {best_fitness}")
    print(f"Set A         : {set_a}")
    print(f"Set B         : {set_b}")

    print("\nConvergence history:")
    print(history)
