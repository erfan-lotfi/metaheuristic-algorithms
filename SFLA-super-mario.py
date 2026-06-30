import random
import math

# ---------------------------------------------------------
# 1. PROBLEM AND ENVIRONMENT DEFINITION (SUPER MARIO GRID)
# ---------------------------------------------------------
# 0: Empty, 1: Coin, 2: Enemy, 3: Wall, 4: Start Position, 5: Flag (Goal)
GRID = [
    [4, 0, 1, 0, 0],
    [3, 3, 0, 3, 0],
    [1, 0, 2, 1, 0],
    [0, 3, 3, 3, 0],
    [0, 0, 1, 2, 5]
]

ROWS = len(GRID)
COLS = len(GRID[0])

# Move mappings: 0 = Up, 1 = Down, 2 = Left, 3 = Right
MOVE_MAP = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

# Find Start and Flag positions dynamically
START_POS = None
FLAG_POS = None
for r in range(ROWS):
    for c in range(COLS):
        if GRID[r][c] == 4:
            START_POS = (r, c)
        elif GRID[r][c] == 5:
            FLAG_POS = (r, c)


# ---------------------------------------------------------
# 2. FITNESS FUNCTION (EVALUATION BASED ON SLIDES LOGIC)
# ---------------------------------------------------------
def evaluate_fitness(position):
    """
    Decodes the continuous frog position into a sequence of moves,
    simulates Mario's path, and computes the fitness score.
    """
    moves = [int(x) for x in position]
    
    current_r, current_c = START_POS
    visited_coins = set()
    enemies_hit = 0
    flag_reached = False
    steps_taken = 0
    
    for move in moves:
        if flag_reached:
            break
        
        steps_taken += 1
        dr, dc = 0, 0
        
        if move == 0: dr = -1   # Up
        elif move == 1: dr = 1  # Down
        elif move == 2: dc = -1 # Left
        elif move == 3: dc = 1  # Right
        
        nr, nc = current_r + dr, current_c + dc
        
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            if GRID[nr][nc] != 3: # Not a wall
                current_r, current_c = nr, nc
                if GRID[current_r][current_c] == 1:
                    visited_coins.add((current_r, current_c))
                elif GRID[current_r][current_c] == 2:
                    enemies_hit += 1
                elif GRID[current_r][current_c] == 5:
                    flag_reached = True
                    
    distance_to_flag = abs(current_r - FLAG_POS[0]) + abs(current_c - FLAG_POS[1])
    
    # Fitness formulation matching the optimization criteria
    fitness = (len(visited_coins) * 20) + \
              (1000 if flag_reached else 0) - \
              (enemies_hit * 30) - \
              (distance_to_flag * 15) - \
              (steps_taken * 0.5)
              
    return fitness


# ---------------------------------------------------------
# 3. FROG CLASS REPRESENTATION
# ---------------------------------------------------------
class Frog:
    def __init__(self, sequence_length, id_num=0):
        self.id = id_num
        self.position = [random.uniform(0.0, 3.99) for _ in range(sequence_length)]
        self.fitness = evaluate_fitness(self.position)
        
    def update_position(self, new_position):
        self.position = [max(0.0, min(3.99, x)) for x in new_position]
        self.fitness = evaluate_fitness(self.position)


# ---------------------------------------------------------
# 4. SHUFFLED FROG LEAPING ALGORITHM (SFLA) CLASS
# ---------------------------------------------------------
class SFLASolver:
    def __init__(self, num_memeplexes, frogs_per_memeplex, local_iterations, global_generations, seq_length, d_max=1.5):
        self.m = num_memeplexes
        self.n = frogs_per_memeplex
        self.pop_size = self.m * self.n
        self.local_iter = local_iterations
        self.global_gen = global_generations
        self.seq_length = seq_length
        self.d_max = d_max
        
        # Initialize population with unique IDs for slide tracking
        self.population = [Frog(self.seq_length, id_num=i+1) for i in range(self.pop_size)]
        self.global_best = None
        
    def run(self):
        print("==========================================================================")
        print("                 SHUFFLED FROG LEAPING ALGORITHM (SFLA)                   ")
        print("==========================================================================")
        print(f"[INIT] Total Population Created: {self.pop_size} frogs.")
        print(f"[INIT] Split configuration: {self.m} Memeplexes x {self.n} Frogs each.")
        print(f"[INIT] Local Search Iterations per Memeplex: {self.local_iter}")
        print("==========================================================================\n")
        
        # Global Shuffling Loop
        for gen in range(1, self.global_gen + 1):
            # 1. Sort the population in descending order of fitness
            self.sort_population()
            self.global_best = self.population[0]
            
            print(f"########### START OF GLOBAL GENERATION {gen} / {self.global_gen} ###########")
            print(f"Current Global Best Fitness (X_g): {self.global_best.fitness:.2f}")
            print("-" * 74)
            
            # 2. Divide population into memeplexes (using the interleaved method from slides)
            memeplexes = [[] for _ in range(self.m)]
            for i, frog in enumerate(self.population):
                memeplexes[i % self.m].append(frog)
                
            # 3. Local Evolution inside each Memeplex
            for mem_idx in range(self.m):
                memeplex = memeplexes[mem_idx]
                print(f"\n  >> Memeplex {mem_idx + 1} Evolution:")
                
                # Show initial state of the memeplex
                print(f"     - Initial Frog IDs : {[f.id for f in memeplex]}")
                print(f"     - Initial Fitnesses: {[round(f.fitness, 1) for f in memeplex]}")
                
                for idx_local in range(1, self.local_iter + 1):
                    X_b = memeplex[0]   # Best frog in current memeplex
                    X_w = memeplex[-1]  # Worst frog in current memeplex
                    X_g = self.global_best
                    
                    old_w_fitness = X_w.fitness
                    
                    # --- Strategy 1: Leap toward Memeplex Best (X_b) ---
                    rand_factor = random.random()
                    D = [rand_factor * (xb - xw) for xb, xw in zip(X_b.position, X_w.position)]
                    D = [max(-self.d_max, min(self.d_max, d_val)) for d_val in D] # Step bounds
                    
                    X_w_new_pos = [xw + d_val for xw, d_val in zip(X_w.position, D)]
                    new_fitness = evaluate_fitness(X_w_new_pos)
                    
                    strategy_used = "Memeplex Best (X_b)"
                    status = "FAILED"
                    
                    if new_fitness > old_w_fitness:
                        X_w.update_position(X_w_new_pos)
                        status = "SUCCESS"
                    else:
                        # --- Strategy 2: Leap toward Global Best (X_g) ---
                        rand_factor = random.random()
                        D = [rand_factor * (xg - xw) for xg, xw in zip(X_g.position, X_w.position)]
                        D = [max(-self.d_max, min(self.d_max, d_val)) for d_val in D]
                        
                        X_w_new_pos = [xw + d_val for xw, d_val in zip(X_w.position, D)]
                        new_fitness = evaluate_fitness(X_w_new_pos)
                        strategy_used = "Global Best (X_g)"
                        
                        if new_fitness > old_w_fitness:
                            X_w.update_position(X_w_new_pos)
                            status = "SUCCESS"
                        else:
                            # --- Strategy 3: Random Replacement ---
                            random_pos = [random.uniform(0.0, 3.99) for _ in range(self.seq_length)]
                            X_w.update_position(random_pos)
                            strategy_used = "Random Generation"
                            status = "FORCED NEW"
                    
                    # Log the exact step behavior like the slide's flowchart trace
                    print(f"     * Local Iter {idx_local}/{self.local_iter} -> Worst Frog (ID: {X_w.id}) updated via {strategy_used} | Status: {status} | Fitness: {old_w_fitness:.1f} -> {X_w.fitness:.1f}")
                    
                    # Re-sort local memeplex after updating the worst frog
                    memeplex.sort(key=lambda f: f.fitness, reverse=True)
            
            # 4. Shuffling Phase
            print("\n" + "-" * 74)
            print(f"[SHUFFLE] Merging all Memeplexes back and Shuffling population...")
            self.population = []
            for memeplex in memeplexes:
                self.population.extend(memeplex)
            print(f"########### END OF GLOBAL GENERATION {gen} ###########\n\n")
            
        # Final Sort to extract the absolute best solution
        self.sort_population()
        self.global_best = self.population[0]
        self.print_final_results()
        
    def sort_population(self):
        self.population.sort(key=lambda frog: frog.fitness, reverse=True)

    def print_final_results(self):
        print("==========================================================================")
        print("                            FINAL SFLA OUTPUT                             ")
        print("==========================================================================")
        print(f"Optimal Fitness Achieved: {self.global_best.fitness:.2f}")
        
        final_moves = [int(x) for x in self.global_best.position]
        move_names = [MOVE_MAP[m] for m in final_moves]
        
        print("\nDecoded Path Sequence:")
        print(" -> ".join(move_names))
        
        # Simulating final run to extract specific environment results
        current_r, current_c = START_POS
        visited_coins = set()
        enemies_hit = 0
        flag_reached = False
        
        for move in final_moves:
            if flag_reached: break
            dr, dc = 0, 0
            if move == 0: dr = -1
            elif move == 1: dr = 1
            elif move == 2: dc = -1
            elif move == 3: dc = 1
            
            nr, nc = current_r + dr, current_c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and GRID[nr][nc] != 3:
                current_r, current_c = nr, nc
                if GRID[current_r][current_c] == 1: visited_coins.add((current_r, current_c))
                elif GRID[current_r][current_c] == 2: enemies_hit += 1
                elif GRID[current_r][current_c] == 5: flag_reached = True
                    
        print("\n--- Game Simulation Metrics ---")
        print(f"Objective Goal Reached? : {'YES! 🚩' if flag_reached else 'NO ❌'}")
        print(f"End Position Coordinate : Row {current_r}, Column {current_c}")
        print(f"Total Unique Coins      : {len(visited_coins)}")
        print(f"Total Danger Collisions : {enemies_hit}")
        print("==========================================================================")


# ---------------------------------------------------------
# 5. EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters setup matching slides criteria
    NUM_MEMEPLEXES = 3
    FROGS_PER_MEMEPLEX = 5
    LOCAL_ITERATIONS = 3
    GLOBAL_GENERATIONS = 10
    SEQUENCE_LENGTH = 15  # Path length (chromosomes length)
    MAX_STEP_SIZE = 1.2   # Maximum change distance (D_max)
    
    solver = SFLASolver(
        num_memeplexes=NUM_MEMEPLEXES,
        frogs_per_memeplex=FROGS_PER_MEMEPLEX,
        local_iterations=LOCAL_ITERATIONS,
        global_generations=GLOBAL_GENERATIONS,
        seq_length=SEQUENCE_LENGTH,
        d_max=MAX_STEP_SIZE
    )
    
    solver.run()