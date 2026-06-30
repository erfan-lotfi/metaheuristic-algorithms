import random
import math

# ---------------------------------------------------------
# 1. ENVIRONMENT & PHYSICS DEFINITION (ANGRY BIRDS GRID)
# ---------------------------------------------------------
GRAVITY = 9.81
TIME_STEP = 0.1  # Simulation time increment
HIT_THRESHOLD = 1.5  # Distance radius to consider a pig "hit"

# Targets to destroy: (x, y) coordinates of the pigs
PIG_POSITIONS = [
    (20.0, 5.0),   # Pig 1 (Close, low)
    (35.0, 12.0),  # Pig 2 (Medium, high up on a structure)
    (50.0, 3.0)    # Pig 3 (Far, low)
]
NUM_PIGS = len(PIG_POSITIONS)

# Gene Bounds for Continuous Search Space
MIN_ANGLE, MAX_ANGLE = 10.0, 85.0     # Launch angle in degrees
MIN_VELOCITY, MAX_VELOCITY = 10.0, 45.0 # Launch velocity in m/s

# ---------------------------------------------------------
# 2. FITNESS FUNCTION WITH TRAJECTORY SIMULATION
# ---------------------------------------------------------
def evaluate_fitness(chromosome):
    """
    Simulates the flight trajectory of 3 consecutive birds.
    Each bird shot is represented by a pair of genes: (angle, velocity).
    """
    # Reshape flat chromosome into 3 distinct shots: [(a1, v1), (a2, v2), (a3, v3)]
    shots = [chromosome[i:i+2] for i in range(0, len(chromosome), 2)]
    
    pigs_alive = [True] * NUM_PIGS
    total_pigs_killed = 0
    min_distance_to_any_pig = float('inf')
    
    for angle, velocity in shots:
        # Convert angle to radians for trigonometry
        rad = math.radians(angle)
        vx = velocity * math.cos(rad)
        vy = velocity * math.sin(rad)
        
        t = 0.0
        while True:
            # Projectile motion equations
            x = vx * t
            y = (vy * t) - (0.5 * GRAVITY * (t ** 2))
            
            # Stop tracking if bird hits the ground or goes too far
            if y < 0 and t > 0:
                break
            if x > 70: 
                break
                
            # Check collision with each alive pig
            for idx, (px, py) in enumerate(PIG_POSITIONS):
                if pigs_alive[idx]:
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    if dist < min_distance_to_any_pig:
                        min_distance_to_any_pig = dist
                        
                    if dist <= HIT_THRESHOLD:
                        pigs_alive[idx] = False
                        total_pigs_killed += 1
            
            t += TIME_STEP
            
    # Fitness Formulation: Big reward for kills, secondary guide via proximity gradient
    proximity_bonus = 100.0 / (1.0 + min_distance_to_any_pig)
    fitness_score = (total_pigs_killed * 1000.0) + proximity_bonus
    
    return fitness_score, total_pigs_killed


# ---------------------------------------------------------
# 3. INDIVIDUAL (CHROMOSOME) REPRESENTATION
# ---------------------------------------------------------
class Individual:
    def __init__(self, num_birds=3, chromosome=None):
        self.num_birds = num_birds
        if chromosome is None:
            # Initialize random continuous values within slide bounds
            self.chromosome = []
            for _ in range(num_birds):
                self.chromosome.append(random.uniform(MIN_ANGLE, MAX_ANGLE))
                self.chromosome.append(random.uniform(MIN_VELOCITY, MAX_VELOCITY))
        else:
            self.chromosome = chromosome
            self.enforce_bounds()
            
        self.fitness, self.pigs_killed = evaluate_fitness(self.chromosome)

    def enforce_bounds(self):
        """Ensures continuous variables do not breach physical simulation limits."""
        for i in range(0, len(self.chromosome), 2):
            self.chromosome[i] = max(MIN_ANGLE, min(MAX_ANGLE, self.chromosome[i]))
            self.chromosome[i+1] = max(MIN_VELOCITY, min(MAX_VELOCITY, self.chromosome[i+1]))


# ---------------------------------------------------------
# 4. GENETIC ALGORITHM (GA) ENGINE
# ---------------------------------------------------------
class GeneticAlgorithmSolver:
    def __init__(self, pop_size, crossover_rate, mutation_rate, generations, num_birds=3):
        self.pop_size = pop_size
        self.pc = crossover_rate
        self.pm = mutation_rate
        self.generations = generations
        self.num_birds = num_birds
        
        # Initialize population
        self.population = [Individual(num_birds=self.num_birds) for _ in range(self.pop_size)]
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        self.best_solution = self.population[0]

    def roulette_wheel_selection(self):
        """Selects a parent based on fitness proportionate selection (Slides Logic)."""
        total_fitness = sum(ind.fitness for ind in self.population)
        
        # Guard against edge case of zero total fitness
        if total_fitness == 0:
            return random.choice(self.population)
            
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in self.population:
            current += ind.fitness
            if current > pick:
                return ind
        return self.population[-1]

    def crossover(self, parent1, parent2):
        """Performs Single-point Crossover structured on structural bird blocks."""
        if random.random() > self.pc:
            return Individual(self.num_birds, list(parent1.chromosome)), Individual(self.num_birds, list(parent2.chromosome))
            
        # Select cross point at a bird boundary (multiples of 2)
        cross_point = random.choice([2, 4])
        
        child1_chrom = parent1.chromosome[:cross_point] + parent2.chromosome[cross_point:]
        child2_chrom = parent2.chromosome[:cross_point] + parent1.chromosome[cross_point:]
        
        return Individual(self.num_birds, child1_chrom), Individual(self.num_birds, child2_chrom)

    def mutate(self, individual):
        """Applies Continuous Gaussian Mutation to angles and velocities."""
        mutated_chrom = list(individual.chromosome)
        mutated = False
        
        for i in range(len(mutated_chrom)):
            if random.random() < self.pm:
                mutated = True
                if i % 2 == 0:
                    # Mutate Angle (Standard deviation of 5 degrees)
                    mutated_chrom[i] += random.gauss(0, 5.0)
                else:
                    # Mutate Velocity (Standard deviation of 3.0 m/s)
                    mutated_chrom[i] += random.gauss(0, 3.0)
                    
        if mutated:
            return Individual(self.num_birds, mutated_chrom)
        return individual

    def run(self):
        print("==========================================================================")
        print("                GENETIC ALGORITHM (GA) FOR ANGRY BIRDS                     ")
        print("==========================================================================")
        print(f"[INIT] Population Size    : {self.pop_size}")
        print(f"[INIT] Crossover Prob (Pc): {self.pc}")
        print(f"[INIT] Mutation Prob (Pm) : {self.pm}")
        print(f"[INIT] Total Pigs Hidden  : {NUM_PIGS} targets at positions {PIG_POSITIONS}")
        print("==========================================================================\n")

        for gen in range(1, self.generations + 1):
            new_population = []
            
            # 1. Elitism: Directly pass the top 2 best solutions to preserve elite genes
            new_population.append(self.population[0])
            new_population.append(self.population[1])
            
            crossover_count = 0
            mutation_count = 0
            
            # 2. Reproduction Loop
            while len(new_population) < self.pop_size:
                # Selection
                p1 = self.roulette_wheel_selection()
                p2 = self.roulette_wheel_selection()
                
                # Crossover
                c1, c2 = self.crossover(p1, p2)
                if c1.chromosome != p1.chromosome: 
                    crossover_count += 1
                
                # Mutation
                m1 = self.mutate(c1)
                m2 = self.mutate(c2)
                if m1 != c1: mutation_count += 1
                if m2 != c2: mutation_count += 1
                
                new_population.append(m1)
                if len(new_population) < self.pop_size:
                    new_population.append(m2)
            
            # Update and rank current generation population
            self.population = new_population
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            if self.population[0].fitness > self.best_solution.fitness:
                self.best_solution = self.population[0]
                
            # Generation structural report mirroring standard course presentation trace
            print(f"########### START OF GENERATION {gen} / {self.generations} ###########")
            print(f" -> Best Fitness in Gen   : {self.population[0].fitness:.2f}")
            print(f" -> Current Kills Logged  : {self.population[0].pigs_killed} / {NUM_PIGS} Pigs")
            print(f" -> Genetic Operations    : Crossovers={crossover_count} | Mutations Appended={mutation_count}")
            
            # Print snapshot of the best trajectory configurations in this generation
            best_chrom = self.population[0].chromosome
            print(" -> Top Chromosome Strategy:")
            for b in range(self.num_birds):
                print(f"     [Bird {b+1}] Angle: {best_chrom[b*2]:.1f}° , Velocity: {best_chrom[b*2+1]:.1f} m/s")
            print(f"########### END OF GENERATION {gen} ###########\n")

        self.print_final_results()

    def print_final_results(self):
        print("==========================================================================")
        print("                            FINAL GA OUTPUT                               ")
        print("==========================================================================")
        print(f"Optimal Solution Fitness : {self.best_solution.fitness:.2f}")
        print(f"Total Pigs Destroyed     : {self.best_solution.pigs_killed} out of {NUM_PIGS}")
        print("\nDecoded Tactical Shots Plan:")
        
        chrom = self.best_solution.chromosome
        for b in range(self.num_birds):
            print(f"  * Slingshot {b+1} -> Launch Angle = {chrom[b*2]:.2f}° | Initial Velocity = {chrom[b*2+1]:.2f} m/s")
        
        print("\nSimulation Verdict       : " + ("MISSION ACCOMPLISHED " if self.best_solution.pigs_killed == NUM_PIGS else "STRATEGY FAILED ❌"))
        print("==========================================================================")


# ---------------------------------------------------------
# 5. EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters from continuous genetic algorithm curriculum standards
    POPULATION_SIZE = 40
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.15
    MAX_GENERATIONS = 15
    TOTAL_BIRDS = 3
    
    solver = GeneticAlgorithmSolver(
        pop_size=POPULATION_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        generations=MAX_GENERATIONS,
        num_birds=TOTAL_BIRDS
    )
    
    solver.run()