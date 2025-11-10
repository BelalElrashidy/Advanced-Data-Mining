import numpy as np
import math

def bpbo(objective,
         dim,
         bounds,
         pop_size=30,
         max_iter=500,
         seed=None,
         verbose=False):
    rng = np.random.default_rng(seed)
    low, high = bounds

    # Step 1: Initialize population (birds) and prey (best)
    population = rng.uniform(low, high, (pop_size, dim))
    fitness = np.array([objective(ind) for ind in population])
    best_idx = np.argmax(fitness)
    prey = population[best_idx].copy()        # current best solution
    prey_fitness = fitness[best_idx]

    fitness_history = [prey_fitness]

    for t in range(1, max_iter + 1):
        # Compute mean of population (for group hunting)
        mean_population = np.mean(population, axis=0)

        new_population = []
        for i in range(pop_size):
            Xi = population[i].copy()

            # --- Individual hunting phase ---
            K1 = rng.integers(1, 3)  # either 1 or 2
            rand_vec = rng.random(dim)
            if K1 == 1:
                # Attack from “behind”
                Xi = Xi + rand_vec * (prey - Xi)
            else:
                # Attack from “front”
                Xi = Xi - rand_vec * (prey - Xi)

            # --- Group hunting phase ---
            K2 = rng.integers(1, 3)
            rand_vec2 = rng.random(dim)
            if K2 == 1:
                Xi = Xi + rand_vec2 * (mean_population - Xi)
            else:
                Xi = Xi - rand_vec2 * (mean_population - Xi)

            # --- Competition / weak-targeting phase ---
            # find worst bird
            worst_idx = int(np.argmin(fitness))
            Xworst = population[worst_idx]
            K3 = rng.integers(1, 3)
            rand_vec3 = rng.random(dim)
            if K3 == 1:
                Xi = Xi + rand_vec3 * (Xworst - Xi)
            else:
                Xi = Xi - rand_vec3 * (Xworst - Xi)

            # --- Reposition / exploration if stuck ---
            # If improvement is poor, random relocate
            # (Use a random jump toward a new random position)
            if rng.random() < 0.1:
                Xi = low + rng.random(dim) * (high - low)

            # Clip
            Xi = np.clip(Xi, low, high)
            new_population.append(Xi)

        # Update population and fitness
        population = np.array(new_population)
        fitness = np.array([objective(ind) for ind in population])

        # Update prey (global best) if any new better
        idx = int(np.argmax(fitness))
        if fitness[idx] > prey_fitness:
            prey = population[idx].copy()
            prey_fitness = fitness[idx]

        fitness_history.append(prey_fitness)

        if verbose and (t % (max_iter // 10) == 0 or t == 1):
            print(f"Iter {t}/{max_iter}, Best = {prey_fitness:.6f}")

    return prey, prey_fitness, fitness_history

# -----------------------------
# Lévy flight generator (Mantegna algorithm)
# -----------------------------
def levy_flight(beta, dim, rng):
    """
    Lévy flight approximation using the Mantegna algorithm.
    """
    sigma = (math.gamma(1 + beta) * math.sin(np.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step
# -----------------------------
# BPBO Algorithm with Lévy adjustment
# -----------------------------
def bpbo_with_levy(objective,
         dim,
         bounds,
         pop_size=30,
         max_iter=500,
         beta=1.5,             # Lévy parameter
        levy_prob=0.1,
         levy_scale=0.05,      # Lévy step scaling factor
         seed=None,
         verbose=False):
    
    rng = np.random.default_rng(seed)
    low, high = bounds

    # Step 1: Initialize population (birds) and prey (best)
    population = rng.uniform(low, high, (pop_size, dim))
    fitness = np.array([objective(ind) for ind in population])
    best_idx = np.argmax(fitness)
    prey = population[best_idx].copy()
    prey_fitness = fitness[best_idx]

    fitness_history = [prey_fitness]

    for t in range(1, max_iter + 1):
        mean_population = np.mean(population, axis=0)
        worst_idx = int(np.argmin(fitness))
        Xworst = population[worst_idx]

        new_population = []

        for i in range(pop_size):
            Xi = population[i].copy()

            # --- Individual hunting phase ---
            K1 = rng.integers(1, 3)
            rand_vec = rng.random(dim)
            if K1 == 1:
                Xi = Xi + rand_vec * (prey - Xi)
            else:
                Xi = Xi - rand_vec * (prey - Xi)

            # --- Group hunting phase ---
            K2 = rng.integers(1, 3)
            rand_vec2 = rng.random(dim)
            if K2 == 1:
                Xi = Xi + rand_vec2 * (mean_population - Xi)
            else:
                Xi = Xi - rand_vec2 * (mean_population - Xi)

            # --- Competition / weak-targeting phase ---
            K3 = rng.integers(1, 3)
            rand_vec3 = rng.random(dim)
            if K3 == 1:
                Xi = Xi + rand_vec3 * (Xworst - Xi)
            else:
                Xi = Xi - rand_vec3 * (Xworst - Xi)

            # --- Lévy flight exploration ---
            if rng.random() < levy_prob:
                Xi += levy_flight(dim=dim, beta=beta, rng=rng) * (high - low)

            # --- Reposition (random reset) ---
            elif rng.random() < 0.05:
                Xi = low + rng.random(dim) * (high - low)

            Xi = np.clip(Xi, low, high)
            new_population.append(Xi)

        # Update population and fitness
        population = np.array(new_population)
        fitness = np.array([objective(ind) for ind in population])

        # Update prey (best)
        idx = np.argmax(fitness)
        if fitness[idx] > prey_fitness:
            prey = population[idx].copy()
            prey_fitness = fitness[idx]

        fitness_history.append(prey_fitness)

        if verbose and (t % (max_iter // 10) == 0 or t == 1):
            print(f"Iter {t}/{max_iter} | Best = {prey_fitness:.6f}")

    return prey, prey_fitness, fitness_history
