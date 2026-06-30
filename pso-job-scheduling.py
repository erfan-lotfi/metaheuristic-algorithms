import random


# Particle class:
# Each particle represents one candidate solution (a job sequence).
# It also stores:
# - velocity: continuous values used by PSO update formula
# - cost: objective value of current position
# - best_position: personal best position (PB)
# - best_cost: objective value of personal best
class Particle:
    def __init__(self, num_jobs):
        # Random initial permutation of jobs
        self.position = random.sample(range(num_jobs), num_jobs)

        # Random initial velocity for each dimension
        self.velocity = [random.uniform(-2, 2) for _ in range(num_jobs)]

        # Current solution cost
        self.cost = float('inf')

        # Personal best position
        self.best_position = []

        # Personal best cost
        self.best_cost = float('inf')


# Objective function:
# Calculate the makespan in a flow shop scheduling problem.
#
# processing_times[job][machine] = processing time of a specific job on a specific machine
# sequence = order of jobs
#
# completion_times[i][m] = completion time of the i-th job in the sequence on machine m
#
# Flow shop rules:
# - Each job must visit machines in the same order
# - A machine can process only one job at a time
# - A job can be processed by only one machine at a time
#
# Makespan = completion time of the last job on the last machine
def calculate_makespan(sequence, processing_times):
    num_jobs = len(sequence)
    num_machines = len(processing_times[0])

    # Create completion time table
    completion_times = [[0 for _ in range(num_machines)] for _ in range(num_jobs)]

    # Fill completion time table
    for i in range(num_jobs):
        job = sequence[i]
        for m in range(num_machines):
            # First job on first machine
            if i == 0 and m == 0:
                completion_times[i][m] = processing_times[job][m]

            # First job on other machines
            elif i == 0:
                completion_times[i][m] = completion_times[i][m - 1] + processing_times[job][m]

            # Other jobs on first machine
            elif m == 0:
                completion_times[i][m] = completion_times[i - 1][m] + processing_times[job][m]

            # General case:
            # A job on machine m starts only when:
            # 1. the same job finishes on machine m-1
            # 2. the previous job finishes on machine m
            else:
                completion_times[i][m] = max(completion_times[i - 1][m], completion_times[i][m - 1]) + processing_times[job][m]

    # Return makespan
    return completion_times[num_jobs - 1][num_machines - 1]


# Evaluation function:
# Returns the objective value of a given sequence.
# Here the objective is the makespan.
def evaluate(sequence, processing_times):
    return calculate_makespan(sequence, processing_times)


# Repair function:
# Since PSO updates the position in a continuous space,
# the resulting values may not form a valid permutation.
#
# This function repairs the position using the same logic as the slides:
# 1. Round values
# 2. Apply lower and upper bounds
# 3. Remove duplicates
# 4. Replace duplicates with missing values
def repair_permutation(position, num_jobs):
    # Step 1: round all values
    repaired = [round(x) for x in position]

    # Step 2: apply bounds to keep values between 0 and num_jobs - 1
    repaired = [min(max(x, 0), num_jobs - 1) for x in repaired]

    # Set for tracking used values
    used = set()

    # Store indices of duplicate values
    duplicate_indices = []

    # Detect duplicates
    for i, val in enumerate(repaired):
        if val in used:
            duplicate_indices.append(i)
        else:
            used.add(val)

    # Find missing job indices
    missing_values = sorted(set(range(num_jobs)) - used)

    # Replace duplicate positions with missing values
    for i, idx in enumerate(duplicate_indices):
        repaired[idx] = missing_values[i]

    return repaired


# Main PSO algorithm for flow shop scheduling
def pso_flow_shop_scheduling(processing_times,
                             num_particles=4,
                             max_iter=50,
                             w=0.5,
                             c1=0.5,
                             c2=1/3,
                             seed=None):
    # If a seed is provided, use it for reproducibility
    if seed is not None:
        random.seed(seed)

    # Number of jobs
    num_jobs = len(processing_times)

    # Create swarm
    swarm = [Particle(num_jobs) for _ in range(num_particles)]

    # Initialize global best (GB)
    global_best_position = None
    global_best_cost = float('inf')

    # Evaluate initial population and initialize PB and GB
    for particle in swarm:
        # Evaluate current particle position
        particle.cost = evaluate(particle.position, processing_times)

        # Initial personal best is the current position
        particle.best_position = particle.position[:]
        particle.best_cost = particle.cost

        # Update global best if needed
        if particle.cost < global_best_cost:
            global_best_cost = particle.cost
            global_best_position = particle.position[:]

    # Print initial population
    print("Initial Population")
    for i, particle in enumerate(swarm, start=1):
        print(f"X{i}(0) = {particle.position}, V{i}(0) = {particle.velocity}, Cost = {particle.cost}")
    print("GBest =", global_best_position, "Cost =", global_best_cost)
    print()

    # Main iteration loop
    for it in range(max_iter):
        print(f"Iteration {it+1}")

        # Update each particle
        for p_idx, particle in enumerate(swarm, start=1):
            new_velocity = []

            # Update velocity for each dimension
            for i in range(num_jobs):
                # Random coefficients
                r1 = random.random()
                r2 = random.random()

                # Inertia component
                inertia = w * particle.velocity[i]

                # Cognitive component: move toward personal best
                cognitive = c1 * r1 * (particle.best_position[i] - particle.position[i])

                # Social component: move toward global best
                social = c2 * r2 * (global_best_position[i] - particle.position[i])

                # PSO velocity update formula
                v_new = inertia + cognitive + social
                new_velocity.append(v_new)

            # Assign new velocity
            particle.velocity = new_velocity

            # Compute raw continuous position
            raw_position = [
                particle.position[i] + particle.velocity[i]
                for i in range(num_jobs)
            ]

            # Repair position to obtain a valid permutation
            repaired_position = repair_permutation(raw_position, num_jobs)

            # Print raw and repaired positions
            print(f"X{p_idx} raw position = {raw_position}")
            print(f"X{p_idx} repaired     = {repaired_position}")

            # Update particle position
            particle.position = repaired_position

            # Evaluate new position
            particle.cost = evaluate(particle.position, processing_times)

            # Update personal best if current solution is better
            if particle.cost < particle.best_cost:
                particle.best_cost = particle.cost
                particle.best_position = particle.position[:]

            # Update global best if current solution is better than current GB
            if particle.cost < global_best_cost:
                global_best_cost = particle.cost
                global_best_position = particle.position[:]

            # Print particle status
            print(f"X{p_idx} new cost     = {particle.cost}")
            print(f"X{p_idx} PB           = {particle.best_position}, PB Cost = {particle.best_cost}")
            print()

        # Print global best after each iteration
        print("GBest =", global_best_position, "Cost =", global_best_cost)
        print("-" * 60)

    # Return final best solution
    return global_best_position, global_best_cost


# Example processing time matrix
# Each row represents one job
# Each column represents one machine
#
# Example:
# Job 0 -> [6, 3, 2]
# means:
# - processing time on machine 1 = 6
# - processing time on machine 2 = 3
# - processing time on machine 3 = 2
processing_times = [
    [6, 3, 2],
    [2, 4, 1],
    [8, 2, 5],
    [3, 5, 2],
    [5, 1, 4]
]

# Run PSO for flow shop scheduling
best_sequence, best_cost = pso_flow_shop_scheduling(
    processing_times=processing_times,
    num_particles=4,
    max_iter=20,
    w=0.5,
    c1=0.5,
    c2=1/3,
    seed=42
)

# Print final result
print("\nFinal Best Sequence:", best_sequence)
print("Final Best Makespan:", best_cost)
