import random


class ABCGridObstacleAvoidance:
    def __init__(
        self,
        grid,
        start,
        goal,
        path_length=10,
        colony_size=8,
        employed_bees=3,
        onlooker_bees=4,
        max_cycles=4,
        limit=4,
        abandon_percent=0.25,
        seed=42
    ):
        """
        Artificial Bee Colony (ABC) for Grid Obstacle Avoidance with Minimal Moves

        Representation:
            Each solution is a vector X of length D = path_length.
            Each element is one move:
                0 = Up
                1 = Down
                2 = Left
                3 = Right

        Objective:
            Reach goal with minimum number of moves while avoiding obstacles.
        """
        random.seed(seed)

        # Problem data
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

        # Representation
        self.D = path_length
        self.Xmin = 0
        self.Xmax = 3  # 4 possible actions

        # ABC parameters
        self.colony_size = colony_size
        self.num_employed = employed_bees
        self.num_onlooker = onlooker_bees
        self.max_cycles = max_cycles
        self.limit = limit
        self.abandon_percent = abandon_percent

        # Food sources
        self.sources = []
        self.F_values = []
        self.fit_values = []
        self.trials = []

        self.move_names = {
            0: "U",
            1: "D",
            2: "L",
            3: "R"
        }

    # --------------------------------------------------
    # Formatting helpers
    # --------------------------------------------------
    def line(self, char="=", n=90):
        print(char * n)

    def title(self, text):
        self.line("=")
        print(text)
        self.line("=")

    def subtitle(self, text):
        self.line("-")
        print(text)
        self.line("-")

    # --------------------------------------------------
    # Utility functions
    # --------------------------------------------------
    def apply_bounds(self, value):
        """Round and wrap into [0,3]."""
        x = int(round(value))
        while x < self.Xmin:
            x = (self.Xmax + 1) + x
        while x > self.Xmax:
            x = x - (self.Xmax + 1)
        return x

    def random_source(self):
        """Generate random sequence of moves."""
        source = []
        for _ in range(self.D):
            r = random.random()
            xij = self.Xmin + r * (self.Xmax - self.Xmin)
            source.append(self.apply_bounds(xij))
        return source

    def move_to_text(self, source):
        return "".join(self.move_names[m] for m in source)

    def is_valid_cell(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def simulate_path(self, source):
        """
        Simulate moves from start position.
        Returns:
            final_position,
            reached_goal,
            valid_steps,
            invalid_steps,
            steps_used,
            path_trace
        """
        r, c = self.start
        path_trace = [(r, c)]
        invalid_steps = 0
        reached_goal = False
        steps_used = 0

        for move in source:
            nr, nc = r, c

            if move == 0:   # Up
                nr -= 1
            elif move == 1: # Down
                nr += 1
            elif move == 2: # Left
                nc -= 1
            elif move == 3: # Right
                nc += 1

            steps_used += 1

            if self.is_valid_cell(nr, nc):
                r, c = nr, nc
            else:
                invalid_steps += 1

            path_trace.append((r, c))

            if (r, c) == self.goal:
                reached_goal = True
                break

        valid_steps = steps_used - invalid_steps
        return (r, c), reached_goal, valid_steps, invalid_steps, steps_used, path_trace

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # --------------------------------------------------
    # Objective and fitness
    # --------------------------------------------------
    def objective_function(self, source):
        """
        Objective:
            If reaches goal:
                F = steps_used + invalid_penalty
            Else:
                F = 100 + distance_to_goal*10 + invalid_penalty + steps_used

        Smaller is better.
        """
        final_pos, reached_goal, _, invalid_steps, steps_used, _ = self.simulate_path(source)
        distance = self.manhattan_distance(final_pos, self.goal)

        if reached_goal:
            return steps_used + invalid_steps * 5
        else:
            return 100 + distance * 10 + invalid_steps * 5 + steps_used

    def fitness_function(self, source):
        fx = self.objective_function(source)
        return 1 / (1 + fx)

    def evaluate_source(self, source):
        fx = self.objective_function(source)
        fitx = 1 / (1 + fx)
        return fx, fitx

    # --------------------------------------------------
    # Table printing
    # --------------------------------------------------
    def print_sources_table(self, title_text="Current Food Sources"):
        self.subtitle(title_text)
        print(f"{'Src':<5}{'Moves':<20}{'F(X)':<10}{'fit(X)':<12}{'Trial':<8}")
        self.line("-", 70)
        for i in range(len(self.sources)):
            move_text = self.move_to_text(self.sources[i])
            print(
                f"{i+1:<5}"
                f"{move_text:<20}"
                f"{self.F_values[i]:<10}"
                f"{self.fit_values[i]:<12.6f}"
                f"{self.trials[i]:<8}"
            )
        self.line("-", 70)

    def print_probabilities_table(self, probabilities):
        self.subtitle("Roulette Wheel Probabilities")
        print(f"{'Src':<5}{'fit(X)':<15}{'P(i)':<15}")
        self.line("-", 40)
        for i, p in enumerate(probabilities):
            print(f"{i+1:<5}{self.fit_values[i]:<15.6f}{p:<15.6f}")
        self.line("-", 40)

    def print_best_summary(self, cycle):
        best_idx = min(range(len(self.sources)), key=lambda i: self.F_values[i])
        best_source = self.sources[best_idx]
        final_pos, reached_goal, valid_steps, invalid_steps, steps_used, path_trace = self.simulate_path(best_source)

        self.subtitle(f"Best Solution After Cycle {cycle}")
        print(f"Best Source Index : {best_idx + 1}")
        print(f"Best Moves        : {self.move_to_text(best_source)}")
        print(f"Best Sequence     : {best_source}")
        print(f"Best F(X)         : {self.F_values[best_idx]}")
        print(f"Best fit(X)       : {self.fit_values[best_idx]:.6f}")
        print(f"Reached Goal      : {reached_goal}")
        print(f"Final Position    : {final_pos}")
        print(f"Valid Steps       : {valid_steps}")
        print(f"Invalid Steps     : {invalid_steps}")
        print(f"Steps Used        : {steps_used}")
        print(f"Path Trace        : {path_trace}")

    def print_grid(self):
        self.subtitle("Grid Layout")
        for r in range(self.rows):
            row_text = ""
            for c in range(self.cols):
                if (r, c) == self.start:
                    row_text += "S "
                elif (r, c) == self.goal:
                    row_text += "G "
                elif self.grid[r][c] == 1:
                    row_text += "# "
                else:
                    row_text += ". "
            print(row_text)

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def initialize(self):
        self.title("STEP 1 & 2: INITIALIZATION OF COLONY AND FOOD SOURCES")

        print(f"Grid size          : {self.rows} x {self.cols}")
        print(f"Start              : {self.start}")
        print(f"Goal               : {self.goal}")
        print(f"Path length (D)    : {self.D}")
        print(f"Move encoding      : 0=U, 1=D, 2=L, 3=R")
        print(f"Colony size        : {self.colony_size}")
        print(f"Employed bees      : {self.num_employed}")
        print(f"Onlooker bees      : {self.num_onlooker}")
        print(f"Limit              : {self.limit}")
        print(f"Max cycles         : {self.max_cycles}")

        self.print_grid()

        self.sources = []
        self.F_values = []
        self.fit_values = []
        self.trials = []

        for _ in range(self.num_employed):
            source = self.random_source()
            fx, fitx = self.evaluate_source(source)

            self.sources.append(source)
            self.F_values.append(fx)
            self.fit_values.append(fitx)
            self.trials.append(0)

        self.print_sources_table("Initial Food Sources")

    # --------------------------------------------------
    # Step 3: Elimination
    # --------------------------------------------------
    def eliminate_weak_sources(self):
        self.title("STEP 3: ELIMINATION OF WEAK SOURCES")

        remove_count = max(1, int(round(self.abandon_percent * len(self.sources))))
        indexed = list(range(len(self.sources)))
        indexed.sort(key=lambda i: self.fit_values[i])
        to_remove = indexed[:remove_count]

        print(f"Elimination rate   : {self.abandon_percent * 100:.0f}%")
        print(f"Removed sources    : {[i + 1 for i in to_remove]}")

        for i in sorted(to_remove, reverse=True):
            del self.sources[i]
            del self.F_values[i]
            del self.fit_values[i]
            del self.trials[i]

        while len(self.sources) < self.num_employed:
            new_source = self.random_source()
            fx, fitx = self.evaluate_source(new_source)

            self.sources.append(new_source)
            self.F_values.append(fx)
            self.fit_values.append(fitx)
            self.trials.append(0)

        self.print_sources_table("Food Sources After Elimination and Replacement")

    # --------------------------------------------------
    # Neighbor generation
    # --------------------------------------------------
    def generate_neighbor(self, i, k):
        current = self.sources[i]
        other = self.sources[k]
        candidate = []

        for j in range(self.D):
            phi = random.uniform(-1, 1)
            new_value = current[j] + phi * (current[j] - other[j])
            candidate.append(self.apply_bounds(new_value))

        return candidate

    # --------------------------------------------------
    # Greedy update
    # --------------------------------------------------
    def greedy_update(self, i, candidate):
        candidate_fx, candidate_fit = self.evaluate_source(candidate)

        if candidate_fit > self.fit_values[i]:
            old_fx = self.F_values[i]
            self.sources[i] = candidate
            self.F_values[i] = candidate_fx
            self.fit_values[i] = candidate_fit
            self.trials[i] = 0
            return True, old_fx, candidate_fx
        else:
            self.trials[i] += 1
            return False, self.F_values[i], candidate_fx

    # --------------------------------------------------
    # Employed phase
    # --------------------------------------------------
    def employed_phase(self):
        self.title("STEP 4: EMPLOYED BEES PHASE")

        print(f"{'Bee':<6}{'Source':<8}{'Partner':<10}{'Old F(X)':<12}{'New F(X)':<12}{'Status':<12}")
        self.line("-", 70)

        for i in range(len(self.sources)):
            choices = [k for k in range(len(self.sources)) if k != i]
            k = random.choice(choices)

            candidate = self.generate_neighbor(i, k)
            improved, old_fx, new_fx = self.greedy_update(i, candidate)
            status = "Improved" if improved else "Rejected"

            print(f"{i+1:<6}{i+1:<8}{k+1:<10}{old_fx:<12}{new_fx:<12}{status:<12}")

        self.line("-", 70)
        self.print_sources_table("Food Sources After Employed Bees Phase")

    # --------------------------------------------------
    # Onlooker phase
    # --------------------------------------------------
    def onlooker_phase(self):
        self.title("STEP 4: ONLOOKER BEES PHASE")

        total_fit = sum(self.fit_values)
        probabilities = [fit / total_fit for fit in self.fit_values]
        self.print_probabilities_table(probabilities)

        print(f"{'Onlooker':<10}{'Chosen Src':<12}{'Partner':<10}{'Old F(X)':<12}{'New F(X)':<12}{'Status':<12}")
        self.line("-", 80)

        for bee in range(self.num_onlooker):
            i = random.choices(range(len(self.sources)), weights=probabilities, k=1)[0]
            choices = [k for k in range(len(self.sources)) if k != i]
            k = random.choice(choices)

            candidate = self.generate_neighbor(i, k)
            improved, old_fx, new_fx = self.greedy_update(i, candidate)
            status = "Improved" if improved else "Rejected"

            print(f"{bee+1:<10}{i+1:<12}{k+1:<10}{old_fx:<12}{new_fx:<12}{status:<12}")

        self.line("-", 80)
        self.print_sources_table("Food Sources After Onlooker Bees Phase")

    # --------------------------------------------------
    # Scout phase
    # --------------------------------------------------
    def scout_phase(self):
        self.title("STEP 5 & 6: ABANDONED SOURCES AND SCOUT BEES")

        abandoned_any = False
        print(f"{'Source':<10}{'Trial Counter':<15}{'Limit':<10}{'Action':<20}")
        self.line("-", 60)

        for i in range(len(self.sources)):
            current_trial = self.trials[i]

            if current_trial >= self.limit:
                abandoned_any = True
                new_source = self.random_source()
                fx, fitx = self.evaluate_source(new_source)

                self.sources[i] = new_source
                self.F_values[i] = fx
                self.fit_values[i] = fitx
                self.trials[i] = 0

                action = "Replaced by scout"
                shown_trial = current_trial
            else:
                action = "Keep source"
                shown_trial = current_trial

            print(f"{i+1:<10}{shown_trial:<15}{self.limit:<10}{action:<20}")

        self.line("-", 60)

        if not abandoned_any:
            print("No source reached the abandonment limit in this cycle.")

        self.print_sources_table("Food Sources After Scout Phase")

    # --------------------------------------------------
    # Run algorithm
    # --------------------------------------------------
    def run(self):
        self.initialize()

        for cycle in range(1, self.max_cycles + 1):
            self.title(f"ABC CYCLE {cycle}")

            self.eliminate_weak_sources()
            self.employed_phase()
            self.onlooker_phase()
            self.scout_phase()
            self.print_best_summary(cycle)

        self.title("FINAL RESULT")
        best_idx = min(range(len(self.sources)), key=lambda i: self.F_values[i])
        best_source = self.sources[best_idx]
        final_pos, reached_goal, valid_steps, invalid_steps, steps_used, path_trace = self.simulate_path(best_source)

        print(f"Best Source Index : {best_idx + 1}")
        print(f"Best Moves        : {self.move_to_text(best_source)}")
        print(f"Best Sequence     : {best_source}")
        print(f"Best F(X)         : {self.F_values[best_idx]}")
        print(f"Best fit(X)       : {self.fit_values[best_idx]:.6f}")
        print(f"Reached Goal      : {reached_goal}")
        print(f"Final Position    : {final_pos}")
        print(f"Steps Used        : {steps_used}")
        print(f"Invalid Steps     : {invalid_steps}")
        print(f"Path Trace        : {path_trace}")

        return best_source, self.F_values[best_idx], self.fit_values[best_idx]


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    # 0 = free cell, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    goal = (4, 4)

    abc = ABCGridObstacleAvoidance(
        grid=grid,
        start=start,
        goal=goal,
        path_length=10,
        colony_size=8,
        employed_bees=3,
        onlooker_bees=4,
        max_cycles=3,
        limit=4,
        abandon_percent=0.25,
        seed=42
    )

    best_solution, best_fx, best_fit = abc.run()

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Start          : {start}")
    print(f"Goal           : {goal}")
    print(f"Best Solution  : {best_solution}")
    print(f"Best Moves     : {''.join(['UDLR'[m] for m in best_solution])}")
    print(f"Best Objective : {best_fx}")
    print(f"Best Fitness   : {best_fit:.6f}")
