import math
import random
from typing import List, Tuple, Optional


class COACirclePacking:
    """
    Cuckoo Optimization Algorithm (COA) for the Circle Packing in a Square problem.
    
    Each habitat represents the coordinates of the centers of N circles inside a [0, 1] x [0, 1] square.
    For N circles, a solution is a continuous vector of length 2*N: [x1, y1, x2, y2, ..., xN, yN].
    
    The profit (fitness) value is the maximum possible radius R that prevents any overlap 
    between circles and keeps all circles entirely within the square boundaries.
    """

    def __init__(
        self,
        num_circles: int,
        n_cuckoos: int = 15,
        max_iter: int = 30,
        min_eggs: int = 2,
        max_eggs: int = 4,
        max_population: int = 30,
        num_clusters: int = 3,
        p_kill: float = 0.2,
        elr_coef: float = 0.4,
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the COA solver for Circle Packing.
        """
        if seed is not None:
            random.seed(seed)

        self.num_circles = num_circles
        self.dim = 2 * num_circles
        self.n_cuckoos = n_cuckoos
        self.max_iter = max_iter
        self.min_eggs = min_eggs
        self.max_eggs = max_eggs
        self.max_population = max_population
        self.num_clusters = num_clusters
        self.p_kill = p_kill
        self.elr_coef = elr_coef
        self.verbose = verbose

        # Search space bounds
        self.var_min = 0.0
        self.var_max = 1.0

    # ------------------------------------------------------------
    # Step 1: Initialize Cuckoo Population (Habitats)
    # ------------------------------------------------------------
    def initialize_population(self) -> List[List[float]]:
        """
        Create the initial random continuous coordinates for the circles.
        """
        population = []
        for _ in range(self.n_cuckoos):
            habitat = [random.uniform(self.var_min, self.var_max) for _ in range(self.dim)]
            population.append(habitat)
        return population

    # ------------------------------------------------------------
    # Step 2: Profit (Fitness) Function for Circle Packing
    # ------------------------------------------------------------
    def profit(self, solution: List[float]) -> float:
        """
        Calculate the maximum possible valid radius R for a given set of centers.
        We want to MAXIMIZE this value.
        """
        # Group flat vector into (x, y) coordinates
        centers = []
        for i in range(0, self.dim, 2):
            centers.append((solution[i], solution[i+1]))

        # 1. Constraint: Distance from each center to the square walls [0, 1]
        # R <= x, R <= 1-x, R <= y, R <= 1-y
        min_boundary_dist = float('inf')
        for x, y in centers:
            dist = min(x, 1.0 - x, y, 1.0 - y)
            if dist < min_boundary_dist:
                min_boundary_dist = dist

        # 2. Constraint: No overlap between any pair of circles
        # dist(C_i, C_j) >= 2R  => R <= dist / 2
        min_pair_dist = float('inf')
        for i in range(self.num_circles):
            for j in range(i + 1, self.num_circles):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_pair_dist:
                    min_pair_dist = dist

        # Valid radius is bound by the strictest condition
        max_r_boundary = min_boundary_dist
        max_r_overlap = min_pair_dist / 2.0

        valid_radius = min(max_r_boundary, max_r_overlap)
        return max(0.0, valid_radius)

    # ------------------------------------------------------------
    # Step 3: Sort Population based on Profit
    # ------------------------------------------------------------
    def sort_population(
        self, 
        population: List[List[float]]
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Sort the habitats in descending order of their profit values.
        """
        scored_population = []
        for habitat in population:
            score = self.profit(habitat)
            scored_population.append((habitat, score))

        # Sort descending (higher profit is better)
        scored_population.sort(key=lambda item: item[1], reverse=True)

        sorted_population = [item[0] for item in scored_population]
        sorted_profits = [item[1] for item in scored_population]

        return sorted_population, sorted_profits

    # ------------------------------------------------------------
    # Step 4: Pure Python K-Means Clustering implementation
    # ------------------------------------------------------------
    def _kmeans_clustering(self, data: List[List[float]], k: int, max_clusters_iter: int = 10) -> List[int]:
        """
        Clusters the cuckoo population into distinct societies without external libraries.
        """
        n = len(data)
        if n == 0:
            return []
        if k >= n:
            return list(range(n))

        # Randomly choose initial centroids from the dataset
        centroid_indices = random.sample(range(n), k)
        centroids = [data[idx][:] for idx in centroid_indices]

        labels = [0] * n
        for _ in range(max_clusters_iter):
            # Assignment step
            for i in range(n):
                min_dist = float('inf')
                best_cluster = 0
                for c_idx in range(k):
                    dist = sum((data[i][d] - centroids[c_idx][d])**2 for d in range(self.dim))
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = c_idx
                labels[i] = best_cluster

            # Update step
            new_centroids = [[0.0] * self.dim for _ in range(k)]
            counts = [0] * k
            for i in range(n):
                c_idx = labels[i]
                counts[c_idx] += 1
                for d in range(self.dim):
                    new_centroids[c_idx][d] += data[i][d]

            for c_idx in range(k):
                if counts[c_idx] > 0:
                    centroids[c_idx] = [new_centroids[c_idx][d] / counts[c_idx] for d in range(self.dim)]
                else:
                    centroids[c_idx] = data[random.randint(0, n - 1)][:]
        return labels

    # ------------------------------------------------------------
    # Step 5: Run the COA Metaheuristic
    # ------------------------------------------------------------
    def run(self) -> Tuple[List[float], float, List[float]]:
        """
        Execute the Cuckoo Optimization Algorithm.
        """
        # Initialize cuckoos randomly
        cuckoos = self.initialize_population()
        cuckoos, cuckoo_profits = self.sort_population(cuckoos)

        # Track the global best habitat configuration
        global_best_habitat = cuckoos[0][: ]
        global_best_profit = cuckoo_profits[0]

        history = [global_best_profit]

        if self.verbose:
            print("Initial population:")
            for i, habitat in enumerate(cuckoos):
                print(f"  Cuckoo {i + 1}: {[round(x, 3) for x in habitat[:4]]}... | profit = {cuckoo_profits[i]:.5f}")
            print("-" * 80)

        # Main Optimization Loop
        for iteration in range(1, self.max_iter + 1):
            if self.verbose:
                print(f"Iteration {iteration}")

            all_eggs = []
            all_eggs_profits = []

            # --- Egg Laying Phase ---
            for i in range(len(cuckoos)):
                # Calculate number of eggs via slide formula range mapping
                num_eggs = int((self.max_eggs - self.min_eggs) * random.random() + self.min_eggs)

                # Calculate distance between current cuckoo and global best habitat
                dist_to_best = math.sqrt(sum((cuckoos[i][d] - global_best_habitat[d])**2 for d in range(self.dim)))
                
                # ELR proportional calculation based on the distance to goal
                elr = self.elr_coef * (dist_to_best + 1e-5)

                if self.verbose:
                    print(f"  Laying Eggs for Cuckoo {i + 1}")
                    print(f"    Current Profit = {cuckoo_profits[i]:.5f} | Target Distance = {dist_to_best:.4f} | ELR = {elr:.4f}")

                # Drop eggs within the Egg Laying Radius (ELR)
                for e in range(num_eggs):
                    egg = []
                    for d in range(self.dim):
                        # Random step inside the localized ELR hyper-sphere
                        random_step = random.uniform(-1, 1)
                        egg_dim_val = cuckoos[i][d] + random_step * elr
                        # Clip to ensure parameters remain inside the environment box boundaries
                        egg_dim_val = max(self.var_min, min(self.var_max, egg_dim_val))
                        egg.append(egg_dim_val)

                    egg_profit = self.profit(egg)
                    all_eggs.append(egg)
                    all_eggs_profits.append(egg_profit)

                    if self.verbose:
                        print(f"    Egg {e + 1} | First 2 dims={[round(egg[0],3), round(egg[1],3)]} | profit = {egg_profit:.5f}")

            # --- Host Bird Detection / Kill Phase ---
            total_laid = len(all_eggs)
            kill_count = int(self.p_kill * total_laid)
            
            # Sort eggs to identify and discard the worst configurations
            scored_eggs = list(zip(all_eggs, all_eggs_profits))
            scored_eggs.sort(key=lambda item: item[1]) # Ascending order (worst first)
            
            # Eliminate detected eggs
            surviving_eggs = [item[0] for item in scored_eggs[kill_count:]]
            surviving_profits = [item[1] for item in scored_eggs[kill_count:]]

            if self.verbose:
                print(f"  Host Bird Detection:")
                print(f"    Total eggs laid: {total_laid} | Eggs eliminated by host: {kill_count}")

            # --- Merging & Population Control ---
            # Young chicks grow up and join the main mature community
            combined_population = cuckoos + surviving_eggs
            cuckoos, cuckoo_profits = self.sort_population(combined_population)

            # Restrict population to max_population to avoid infinite resource explosion
            if len(cuckoos) > self.max_population:
                cuckoos = cuckoos[:self.max_population]
                cuckoo_profits = cuckoo_profits[:self.max_population]

            # Update the absolute global best tracker
            if cuckoo_profits[0] > global_best_profit:
                global_best_profit = cuckoo_profits[0]
                global_best_habitat = cuckoos[0][:]

            # --- Clustering Phase (Society Grouping) ---
            current_k = min(self.num_clusters, len(cuckoos))
            cluster_labels = self._kmeans_clustering(cuckoos, current_k)

            # Find the elite cluster with the highest average profit performance
            best_cluster_idx = 0
            max_cluster_avg = -float('inf')

            for c_id in range(current_k):
                cluster_profits = [cuckoo_profits[idx] for idx in range(len(cuckoos)) if cluster_labels[idx] == c_id]
                if cluster_profits:
                    avg_p = sum(cluster_profits) / len(cluster_profits)
                    if avg_p > max_cluster_avg:
                        max_cluster_avg = avg_p
                        best_cluster_idx = c_id

            # Identify target habitat within the best society group
            best_cluster_indices = [idx for idx in range(len(cuckoos)) if cluster_labels[idx] == best_cluster_idx]
            best_cluster_indices.sort(key=lambda idx: cuckoo_profits[idx], reverse=True)
            target_destination = cuckoos[best_cluster_indices[0]]

            # --- Migration Phase ---
            if self.verbose:
                print(f"  Societies Clustering & Migration:")
                print(f"    Total active clusters: {current_k} | Top group average profit: {max_cluster_avg:.5f}")

            for i in range(len(cuckoos)):
                # Skip moving if it's already the apex target destination
                if cuckoos[i] == target_destination:
                    continue

                # Vectorized movement towards the goal habitat with exploration steps
                new_habitat = []
                for d in range(self.dim):
                    rand_factor = random.random()
                    step = rand_factor * (target_destination[d] - cuckoos[i][d])
                    new_val = max(self.var_min, min(self.var_max, cuckoos[i][d] + step))
                    new_habitat.append(new_val)

                cuckoos[i] = new_habitat
                cuckoo_profits[i] = self.profit(new_habitat)

            # Re-sort and save history metrics after updates
            cuckoos, cuckoo_profits = self.sort_population(cuckoos)
            if cuckoo_profits[0] > global_best_profit:
                global_best_profit = cuckoo_profits[0]
                global_best_habitat = cuckoos[0][:]

            history.append(global_best_profit)

            if self.verbose:
                print("  Best habitat after iteration:")
                print(f"    Best solution centers (first 2) = {[round(global_best_habitat[0],4), round(global_best_habitat[1],4)]}")
                print(f"    Best Packed Radius              = {global_best_profit:.6f}")
                print("-" * 80)

        return global_best_habitat, global_best_profit, history


# --------------------------------------------------------------------
# Example Execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Packing N circles into a [0,1] square
    NUM_CIRCLES_TO_PACK = 5

    # Initialize the COA solver
    solver = COACirclePacking(
        num_circles=NUM_CIRCLES_TO_PACK,
        n_cuckoos=6,          # Small number of cuckoos for clean logs
        max_iter=15,          # Number of generation steps
        min_eggs=2,
        max_eggs=4,
        max_population=15,
        num_clusters=2,       # Grouping into 2 separate societies
        p_kill=0.25,          # 25% probability of host bird discovering eggs
        elr_coef=0.4,
        seed=100,             # Fixed seed for perfect simulation output reproducibility
        verbose=True
    )

    # Run algorithm
    best_packing, optimal_r, convergence_history = solver.run()

    print("\n" + "="*30 + " FINAL PACKING OUTCOME " + "="*30)
    print(f"Optimal Radius (R) Found: {optimal_r:.6f}")
    print("Coordinates of Circle Centers:")
    
    circle_idx = 1
    for idx in range(0, len(best_packing), 2):
        print(f"  Circle {circle_idx:02d} -> X: {best_packing[idx]:.5f}, Y: {best_packing[idx+1]:.5f}")
        circle_idx += 1

    print("\nConvergence Progression History Matrix:")
    print([round(h, 5) for h in convergence_history])