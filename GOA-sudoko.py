import math
import random
import copy


class GrasshopperSudokuSolverVerbose:
    def __init__(
        self,
        puzzle,
        subgrid_size,
        population_size=20,
        max_epochs=100,
        c_min=1e-5,
        c_max=1.0,
        f=0.5,
        l=1.5,
        seed=None,
        show_population_rows=10
    ):
        """
        puzzle: 2D Sudoku grid, empty cells are 0
        subgrid_size: n, where board size is n^2 x n^2
        show_population_rows: number of grasshoppers to print in each epoch table
        """
        if seed is not None:
            random.seed(seed)

        self.puzzle = copy.deepcopy(puzzle)
        self.n = subgrid_size
        self.N = subgrid_size * subgrid_size

        self.population_size = population_size
        self.max_epochs = max_epochs
        self.c_min = c_min
        self.c_max = c_max
        self.f = f
        self.l = l
        self.show_population_rows = show_population_rows

        self.lb = 1
        self.ub = self.N

        self.empty_positions = []
        self.fixed_positions = []

        for r in range(self.N):
            for c in range(self.N):
                if self.puzzle[r][c] == 0:
                    self.empty_positions.append((r, c))
                else:
                    self.fixed_positions.append((r, c))

        self.dimension = len(self.empty_positions)

        self.best_position = None
        self.best_fitness = float("inf")

    def s_function(self, r):
        """
        Social interaction function from GOA slides:
        s(r) = f * exp(-r / l) - exp(-r)
        """
        return self.f * math.exp(-r / self.l) - math.exp(-r)

    def clamp(self, x, low, high):
        return max(low, min(high, x))

    def continuous_to_discrete(self, x):
        """
        Map a continuous value to a valid Sudoku integer in [1, N]
        """
        x = round(x)
        x = int(self.clamp(x, self.lb, self.ub))
        return x

    def build_grid_from_position(self, position):
        """
        Fill puzzle empty cells using a grasshopper position vector
        """
        grid = copy.deepcopy(self.puzzle)
        for idx, (r, c) in enumerate(self.empty_positions):
            grid[r][c] = self.continuous_to_discrete(position[idx])
        return grid

    def count_duplicates(self, values):
        """
        Count duplicate violations in a list
        """
        freq = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1

        violations = 0
        for count in freq.values():
            if count > 1:
                violations += count - 1
        return violations

    def evaluate_grid(self, grid):
        """
        Fitness = total number of violations in rows, columns, and subgrids
        """
        violations = 0

        # Row violations
        for r in range(self.N):
            violations += self.count_duplicates(grid[r])

        # Column violations
        for c in range(self.N):
            col = [grid[r][c] for r in range(self.N)]
            violations += self.count_duplicates(col)

        # Subgrid violations
        for br in range(0, self.N, self.n):
            for bc in range(0, self.N, self.n):
                block = []
                for r in range(br, br + self.n):
                    for c in range(bc, bc + self.n):
                        block.append(grid[r][c])
                violations += self.count_duplicates(block)

        return violations

    def fitness(self, position):
        """
        Compute fitness of one grasshopper
        """
        grid = self.build_grid_from_position(position)
        return self.evaluate_grid(grid)

    def initialize_population(self):
        """
        Create initial population
        """
        population = []
        for _ in range(self.population_size):
            grasshopper = [
                random.uniform(self.lb, self.ub)
                for _ in range(self.dimension)
            ]
            population.append(grasshopper)
        return population

    def update_c(self, epoch):
        """
        c = c_max - epoch * ((c_max - c_min) / max_epochs)
        """
        return self.c_max - epoch * ((self.c_max - self.c_min) / self.max_epochs)

    def discrete_vector(self, position):
        """
        Return discrete version of a grasshopper vector
        """
        return [self.continuous_to_discrete(x) for x in position]

    def print_grid(self, grid):
        """
        Pretty print Sudoku grid
        """
        for row in grid:
            print(" ".join(str(x) for x in row))

    def print_initial_info(self):
        """
        Print initial problem settings similar to a lecture/demo style
        """
        print("=" * 90)
        print("GENERALIZED SUDOKU SOLVING WITH GRASSHOPPER OPTIMIZATION ALGORITHM (GOA)")
        print("=" * 90)
        print(f"Board size                : {self.N} x {self.N}")
        print(f"Subgrid size              : {self.n} x {self.n}")
        print(f"Population size           : {self.population_size}")
        print(f"Max epochs                : {self.max_epochs}")
        print(f"c_min                     : {self.c_min}")
        print(f"c_max                     : {self.c_max}")
        print(f"f                         : {self.f}")
        print(f"l                         : {self.l}")
        print(f"Lower bound               : {self.lb}")
        print(f"Upper bound               : {self.ub}")
        print(f"Number of empty cells     : {self.dimension}")
        print("=" * 90)
        print("Initial puzzle:")
        self.print_grid(self.puzzle)
        print("=" * 90)

    def print_population_table(self, population, title, limit=None):
        """
        Print a table of grasshoppers similar to slide-style examples
        """
        if limit is None:
            limit = self.show_population_rows

        print(title)
        print("-" * 90)
        print(f"{'Idx':<6}{'Grasshopper Position (discrete)':<60}{'Fitness':<10}")
        print("-" * 90)

        rows_to_show = min(limit, len(population))
        for i in range(rows_to_show):
            discrete_pos = self.discrete_vector(population[i])
            fit = self.fitness(population[i])
            print(f"{i+1:<6}{str(discrete_pos):<60}{fit:<10}")

        if len(population) > rows_to_show:
            print("...")
        print("-" * 90)

    def print_best_table(self, epoch, c_value):
        """
        Print the best grasshopper T in table format
        """
        discrete_best = self.discrete_vector(self.best_position)
        print("Best grasshopper (Target T)")
        print("-" * 90)
        print(f"{'Epoch':<10}{'c':<15}{'Best Position (discrete)':<50}{'Best Fitness':<10}")
        print("-" * 90)
        print(f"{epoch:<10}{c_value:<15.7f}{str(discrete_best):<50}{self.best_fitness:<10}")
        print("-" * 90)

    def solve(self, verbose=True):
        """
        Solve Sudoku with GOA and print epoch-by-epoch output
        """
        if self.dimension == 0:
            if verbose:
                self.print_initial_info()
                print("Puzzle has no empty cells.")
                print("Fitness:", self.evaluate_grid(self.puzzle))
            return self.puzzle, self.evaluate_grid(self.puzzle)

        if verbose:
            self.print_initial_info()

        # Step 1: Initialize population
        population = self.initialize_population()

        if verbose:
            print("STEP 1: Generate initial grasshopper population")
            self.print_population_table(
                population,
                title="Initial population of grasshoppers"
            )

        # Step 2: Evaluate population and choose T
        if verbose:
            print("STEP 2: Evaluate the grasshopper population and choose T")

        for i in range(self.population_size):
            fit = self.fitness(population[i])
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_position = population[i][:]

        if verbose:
            self.print_best_table(epoch=0, c_value=self.c_max)

        # Main loop
        for epoch in range(1, self.max_epochs + 1):
            print("\n" + "=" * 90)
            print(f"EPOCH {epoch}")
            print("=" * 90)

            # Step 3: Update c
            c = self.update_c(epoch)
            if verbose:
                print("STEP 3: Update parameter c")
                print(f"c = c_max - epoch * ((c_max - c_min) / max_epochs)")
                print(f"c = {self.c_max} - {epoch} * (({self.c_max} - {self.c_min}) / {self.max_epochs})")
                print(f"c = {c:.7f}")
                print("-" * 90)

            # Step 4: Update grasshopper positions
            if verbose:
                print("STEP 4: Update grasshopper positions using modified GOA equation")

            new_population = []

            for i in range(self.population_size):
                Xi = population[i]
                new_Xi = [0.0] * self.dimension

                for d in range(self.dimension):
                    social_sum = 0.0

                    for j in range(self.population_size):
                        if i == j:
                            continue

                        Xj = population[j]
                        distance_ij = abs(Xj[d] - Xi[d])

                        if distance_ij == 0:
                            direction = 0.0
                        else:
                            direction = (Xj[d] - Xi[d]) / distance_ij

                        social_sum += (
                            c * ((self.ub - self.lb) / 2.0) *
                            self.s_function(distance_ij) *
                            direction
                        )

                    # Modified GOA position update from slides:
                    # X_i^d(t+1) = c * social_sum + T_d
                    new_value = c * social_sum + self.best_position[d]
                    new_value = self.clamp(new_value, self.lb, self.ub)
                    new_Xi[d] = new_value

                new_population.append(new_Xi)

            if verbose:
                self.print_population_table(
                    new_population,
                    title="Updated grasshopper population"
                )

            # Step 5: Evaluate population and update T
            if verbose:
                print("STEP 5: Evaluate updated population and update T if needed")

            population = new_population

            for i in range(self.population_size):
                fit = self.fitness(population[i])
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_position = population[i][:]

            if verbose:
                self.print_best_table(epoch=epoch, c_value=c)

            # Optional: print current best Sudoku grid
            if verbose:
                print("Current best Sudoku grid")
                print("-" * 90)
                best_grid = self.build_grid_from_position(self.best_position)
                self.print_grid(best_grid)
                print("-" * 90)

            # Step 6: Check stopping condition
            if verbose:
                print("STEP 6: Check stopping condition")
                print(f"Best fitness = {self.best_fitness}")
                if self.best_fitness == 0:
                    print("Stopping condition satisfied: Fitness = 0")
                else:
                    print("Stopping condition not satisfied yet")
                print("-" * 90)

            if self.best_fitness == 0:
                break

        best_grid = self.build_grid_from_position(self.best_position)

        if verbose:
            print("\n" + "=" * 90)
            print("FINAL RESULT")
            print("=" * 90)
            print("Best grasshopper position:")
            print(self.discrete_vector(self.best_position))
            print(f"Best fitness: {self.best_fitness}")
            print("Best Sudoku solution found:")
            self.print_grid(best_grid)
            print("=" * 90)

        return best_grid, self.best_fitness


if __name__ == "__main__":
    # Example generalized Sudoku: n=2 => 4x4 board
    puzzle_4x4 = [
        [1, 0, 0, 4],
        [0, 4, 1, 0],
        [2, 0, 4, 3],
        [0, 3, 0, 1]
    ]

    solver = GrasshopperSudokuSolverVerbose(
        puzzle=puzzle_4x4,
        subgrid_size=2,
        population_size=20,
        max_epochs=30,
        c_min=1e-5,
        c_max=1.0,
        f=0.5,
        l=1.5,
        seed=42,
        show_population_rows=10
    )

    solution, fitness = solver.solve(verbose=True)
