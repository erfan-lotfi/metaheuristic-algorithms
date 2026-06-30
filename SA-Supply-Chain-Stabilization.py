import math
import random
import copy

# =========================================================
# Supply Chain Stabilization using Simulated Annealing (SA)
# Step-by-step output exactly in slide style:
# E1, E2, DeltaE, P, r, accepted/rejected
# =========================================================

# -----------------------------
# Problem Parameters
# -----------------------------

# Number of supply chain levels
# 0 = Retailer, 1 = Distributor, 2 = Wholesaler, 3 = Factory
NUM_ECHELONS = 4

# Number of time periods
NUM_PERIODS = 12

# External customer demand at retailer level
CUSTOMER_DEMAND = [20, 25, 18, 30, 28, 22, 35, 27, 24, 29, 31, 26]

# Lead time for each echelon
LEAD_TIME = [1, 2, 2, 3]

# Initial inventory for each echelon
INITIAL_INVENTORY = [40, 50, 60, 80]

# Initial backlog for each echelon
INITIAL_BACKLOG = [0, 0, 0, 0]

# Holding cost for each echelon
HOLDING_COST = [1.0, 0.8, 0.7, 0.6]

# Shortage/backlog cost for each echelon
SHORTAGE_COST = [5.0, 4.5, 4.0, 3.5]

# Penalty for order changes to reduce instability
ORDER_CHANGE_PENALTY = [0.8, 0.7, 0.6, 0.5]

# Maximum order allowed
MAX_ORDER = 60


# -----------------------------
# SA Parameters from Slides
# -----------------------------

INITIAL_TEMPERATURE = 100
FINAL_TEMPERATURE = 0
ITERATIONS_PER_TEMPERATURE = 1
BOLTZMANN_K = 1
COOLING_STEP = 5


# =========================================================
# Generate Initial Solution
# solution[e][t] = order quantity for echelon e at period t
# =========================================================
def generate_initial_solution():
    solution = []
    for e in range(NUM_ECHELONS):
        row = []
        for t in range(NUM_PERIODS):
            row.append(random.randint(0, MAX_ORDER))
        solution.append(row)
    return solution


# =========================================================
# Energy Function E
# E = holding cost + shortage cost + order instability cost
# =========================================================
def simulate_supply_chain(solution):
    inventory = INITIAL_INVENTORY[:]
    backlog = INITIAL_BACKLOG[:]

    # Pipeline to model delayed arrivals
    pipeline = []
    for e in range(NUM_ECHELONS):
        pipeline.append([0] * (NUM_PERIODS + max(LEAD_TIME) + 5))

    total_holding_cost = 0.0
    total_shortage_cost = 0.0
    total_order_change_cost = 0.0

    previous_orders = [0] * NUM_ECHELONS

    for t in range(NUM_PERIODS):

        # Receive delayed shipments
        for e in range(NUM_ECHELONS):
            inventory[e] += pipeline[e][t]

        # Demand at each echelon
        echelon_demand = [0] * NUM_ECHELONS
        echelon_demand[0] = CUSTOMER_DEMAND[t]

        # Upstream demand equals downstream placed order
        for e in range(1, NUM_ECHELONS):
            echelon_demand[e] = solution[e - 1][t]

        # Serve demand and update backlog
        for e in range(NUM_ECHELONS):
            effective_demand = echelon_demand[e] + backlog[e]
            shipped = min(inventory[e], effective_demand)
            inventory[e] -= shipped
            backlog[e] = effective_demand - shipped

        # Place orders and send them into pipeline
        for e in range(NUM_ECHELONS):
            order_qty = solution[e][t]
            arrival_time = t + LEAD_TIME[e]

            if arrival_time < len(pipeline[e]):
                pipeline[e][arrival_time] += order_qty

            # Penalize sudden order changes
            total_order_change_cost += ORDER_CHANGE_PENALTY[e] * abs(order_qty - previous_orders[e])
            previous_orders[e] = order_qty

        # Add holding and shortage costs
        for e in range(NUM_ECHELONS):
            total_holding_cost += HOLDING_COST[e] * inventory[e]
            total_shortage_cost += SHORTAGE_COST[e] * backlog[e]

    total_energy = total_holding_cost + total_shortage_cost + total_order_change_cost

    details = {
        "holding_cost": total_holding_cost,
        "shortage_cost": total_shortage_cost,
        "order_change_cost": total_order_change_cost
    }

    return total_energy, details


# =========================================================
# Neighborhood Structures from slide style
# 1) Swap
# 2) Reversion
# 3) Insertion
# =========================================================
def swap_neighbor(solution):
    new_solution = copy.deepcopy(solution)
    e = random.randint(0, NUM_ECHELONS - 1)
    i, j = random.sample(range(NUM_PERIODS), 2)
    new_solution[e][i], new_solution[e][j] = new_solution[e][j], new_solution[e][i]
    return new_solution, f"Swap in echelon {e}, positions {i} and {j}"


def reversion_neighbor(solution):
    new_solution = copy.deepcopy(solution)
    e = random.randint(0, NUM_ECHELONS - 1)
    i, j = sorted(random.sample(range(NUM_PERIODS), 2))
    new_solution[e][i:j+1] = list(reversed(new_solution[e][i:j+1]))
    return new_solution, f"Reversion in echelon {e}, segment [{i}:{j}]"


def insertion_neighbor(solution):
    new_solution = copy.deepcopy(solution)
    e = random.randint(0, NUM_ECHELONS - 1)
    i, j = random.sample(range(NUM_PERIODS), 2)
    value = new_solution[e].pop(i)
    new_solution[e].insert(j, value)
    return new_solution, f"Insertion in echelon {e}, move from {i} to {j}"


def generate_neighbor(solution):
    move_type = random.choice(["swap", "reversion", "insertion"])

    if move_type == "swap":
        return swap_neighbor(solution)
    elif move_type == "reversion":
        return reversion_neighbor(solution)
    else:
        return insertion_neighbor(solution)


# =========================================================
# Acceptance Criterion Exactly Like Slides
# If DeltaE <= 0 => accept
# Else P = exp(-DeltaE / (K*T)) and compare with r
# =========================================================
def acceptance_decision(E1, E2, temperature):
    delta_e = E2 - E1

    # Better or equal solution is always accepted
    if delta_e <= 0:
        return {
            "delta_e": delta_e,
            "probability": 1.0,
            "random_r": None,
            "accepted": True,
            "reason": "DeltaE <= 0"
        }

    # Worse solution may be accepted probabilistically
    if temperature <= 0:
        return {
            "delta_e": delta_e,
            "probability": 0.0,
            "random_r": None,
            "accepted": False,
            "reason": "Temperature <= 0"
        }

    probability = math.exp(-delta_e / (BOLTZMANN_K * temperature))
    r = random.random()
    accepted = probability > r

    return {
        "delta_e": delta_e,
        "probability": probability,
        "random_r": r,
        "accepted": accepted,
        "reason": "P > r" if accepted else "P <= r"
    }


# =========================================================
# Print Step Like Slides
# =========================================================
def print_step(step_no, temperature, move_description, E1, E2, decision):
    print("=" * 70)
    print(f"Step: {step_no}")
    print(f"Temperature (T): {temperature}")
    print(f"Neighbor Move: {move_description}")
    print(f"E1 (Current Energy): {E1:.2f}")
    print(f"E2 (New Energy):     {E2:.2f}")
    print(f"DeltaE = E2 - E1 = {decision['delta_e']:.2f}")

    if decision["delta_e"] <= 0:
        print("Since DeltaE <= 0, the new solution is accepted directly.")
        print("Accepted: YES")
    else:
        print(f"P = exp(-DeltaE / (K*T)) = exp(-{decision['delta_e']:.2f} / ({BOLTZMANN_K}*{temperature}))")
        print(f"P = {decision['probability']:.6f}")
        print(f"r = {decision['random_r']:.6f}")

        if decision["accepted"]:
            print("Condition: P > r")
            print("Accepted: YES")
        else:
            print("Condition: P <= r")
            print("Accepted: NO")


# =========================================================
# Simulated Annealing Main Function
# =========================================================
def simulated_annealing():
    current_solution = generate_initial_solution()
    current_energy, current_details = simulate_supply_chain(current_solution)

    best_solution = copy.deepcopy(current_solution)
    best_energy = current_energy
    best_details = current_details

    temperature = INITIAL_TEMPERATURE
    step_no = 1

    print("\n" + "#" * 70)
    print("Initial Solution Energy")
    print(f"E = {current_energy:.2f}")
    print(f"Cost Details = {current_details}")
    print("#" * 70 + "\n")

    while temperature > FINAL_TEMPERATURE:
        for _ in range(ITERATIONS_PER_TEMPERATURE):
            new_solution, move_description = generate_neighbor(current_solution)
            new_energy, new_details = simulate_supply_chain(new_solution)

            decision = acceptance_decision(current_energy, new_energy, temperature)

            # Print step-by-step output like slides
            print_step(
                step_no=step_no,
                temperature=temperature,
                move_description=move_description,
                E1=current_energy,
                E2=new_energy,
                decision=decision
            )

            # Accept or reject
            if decision["accepted"]:
                current_solution = new_solution
                current_energy = new_energy
                current_details = new_details

                if current_energy < best_energy:
                    best_solution = copy.deepcopy(current_solution)
                    best_energy = current_energy
                    best_details = current_details

            step_no += 1

        # Linear cooling exactly like slides
        print(f"\nTemperature Update: T(new) = T(old) - 5 = {temperature} - 5 = {temperature - COOLING_STEP}")
        print("\n")
        temperature -= COOLING_STEP

    return best_solution, best_energy, best_details


# =========================================================
# Print Final Best Solution
# =========================================================
def print_best_solution(solution, best_energy, best_details):
    echelon_names = ["Retailer", "Distributor", "Wholesaler", "Factory"]

    print("\n" + "#" * 70)
    print("Final Best Solution")
    print("#" * 70)
    print(f"Best Energy = {best_energy:.2f}")
    print(f"Best Cost Details = {best_details}")
    print()

    for e in range(NUM_ECHELONS):
        print(f"{echelon_names[e]} Orders: {solution[e]}")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    best_solution, best_energy, best_details = simulated_annealing()
    print_best_solution(best_solution, best_energy, best_details)
