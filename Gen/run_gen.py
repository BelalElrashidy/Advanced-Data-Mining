import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
import random
import math
from Gen import rastrigin, roulette_wheel_select, tournament_select, single_point_crossover, two_point_crossover, gaussian_mutation
def run_ga(d=50, pop_size=100, generations=1000, crossover_rate=0.9, mutation_prob_gene=1/50,
           sigma=0.1, selection='tournament', tournament_k=3, crossover='single',
           elitism=1, bounds=(-5.12,5.12), seed=None, verbose=False):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    # init
    pop = np.random.uniform(bounds[0], bounds[1], size=(pop_size, d))
    fitness = rastrigin(pop)
    best_per_gen = []
    best_idx = np.argmin(fitness)
    best_per_gen.append(fitness[best_idx])
    for gen in range(1, generations+1):
        new_pop = []
        # elitism
        elite_count = min(elitism, pop_size)
        elite_idxs = np.argsort(fitness)[:elite_count]
        for idx in elite_idxs:
            new_pop.append(pop[idx].copy())
        # selection of parents for rest
        n_offspring_needed = pop_size - elite_count
        parent_indices = []
        if selection == 'roulette':
            parent_indices = roulette_wheel_select(pop, fitness, n_offspring_needed)
        elif selection == 'tournament':
            parent_indices = tournament_select(pop, fitness, n_offspring_needed, k=tournament_k)
        elif selection == 'hybrid':
            # half tournament half roulette
            half = n_offspring_needed//2
            p1 = tournament_select(pop, fitness, half, k=tournament_k)
            p2 = roulette_wheel_select(pop, fitness, n_offspring_needed - half)
            parent_indices = np.concatenate([p1,p2])
        else:
            parent_indices = tournament_select(pop, fitness, n_offspring_needed, k=tournament_k)
        # create offspring by pairing
        np.random.shuffle(parent_indices)
        i = 0
        while len(new_pop) < pop_size:
            if i+1 < len(parent_indices):
                a = pop[parent_indices[i]]
                b = pop[parent_indices[i+1]]
            else:
                a = pop[parent_indices[i]]
                b = pop[np.random.randint(pop_size)]
            i += 2
            if np.random.rand() < crossover_rate:
                if crossover == 'single':
                    c1, c2 = single_point_crossover(a, b)
                elif crossover == 'two':
                    c1, c2 = two_point_crossover(a, b)
                else:
                    c1, c2 = single_point_crossover(a, b)
            else:
                c1, c2 = a.copy(), b.copy()
            # mutation
            c1 = gaussian_mutation(c1, mutation_prob_gene, sigma, bounds)
            if len(new_pop) < pop_size:
                new_pop.append(c1)
            if len(new_pop) < pop_size:
                c2 = gaussian_mutation(c2, mutation_prob_gene, sigma, bounds)
                new_pop.append(c2)
        pop = np.array(new_pop)[:pop_size]
        fitness = rastrigin(pop)
        best_per_gen.append(np.min(fitness))
        if verbose and gen % 100 == 0:
            print(f"Gen {gen}, best {best_per_gen[-1]:.6f}")
    return np.array(best_per_gen), pop[np.argmin(fitness)], np.min(fitness)

if __name__ == "__main__":
    crossover = ['single', 'two']
    selection = ['roulette', 'tournament', 'hybrid']
    histories = []
    # Example run
    for sel in selection:
        for cro in crossover:
            print(f"Running GA with selection={sel}, crossover={cro}")
            best_curve, best_sol, best_val = run_ga(d=50, pop_size=100, generations=500,
                                           selection=sel, tournament_k=3,
                                           crossover=cro, mutation_prob_gene=1/50, sigma=0.1,
                                           seed=42)
            histories.append((sel, cro, best_curve))
            print("Final best:", best_val)
    # best_curve, best_sol, best_val = run_ga(d=50, pop_size=100, generations=500,
    #                                        selection='tournament', tournament_k=3,
    #                                        crossover='two', mutation_prob_gene=1/50, sigma=0.1,
    #                                        seed=42)
    print("Final best:", best_val)
    # Plot
    for sel, cro, history in histories:
        plt.plot(history, label=f"{sel}-{cro}")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Best fitness (log)')
    plt.title('Convergence (single run)')
    plt.show()
