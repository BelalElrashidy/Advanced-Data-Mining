import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
import random
import math

def rastrigin(pop):
    # pop: (n, d)
    A = 10.0
    d = pop.shape[1]
    return A*d + np.sum(pop**2 - A * np.cos(2*math.pi*pop), axis=1)
def roulette_wheel_select(pop, fitness, n_select):
    # fitness is to minimize; convert to positive scores
    # use inverted ranking to be robust
    ranks = np.argsort(np.argsort(fitness))  # 0 = best
    scores = (len(fitness)-1 - ranks) + 1e-9
    probs = scores / scores.sum()
    idx = np.random.choice(len(pop), size=n_select, replace=True, p=probs)
    return idx
def tournament_select(pop, fitness, n_select, k=3):
    idxs = []
    N = len(pop)
    for _ in range(n_select):
        participants = np.random.choice(N, size=k, replace=False)
        winner = participants[np.argmin(fitness[participants])]
        idxs.append(winner)
    return np.array(idxs)

def single_point_crossover(p1, p2):
    d = p1.shape[0]
    if d <= 1:
        return p1.copy(), p2.copy()
    p = np.random.randint(1, d)
    c1 = np.concatenate([p1[:p], p2[p:]])
    c2 = np.concatenate([p2[:p], p1[p:]])
    return c1, c2
def two_point_crossover(p1, p2):
    d = p1.shape[0]
    a, b = sorted(np.random.choice(range(1,d), size=2, replace=False))
    c1 = np.concatenate([p1[:a], p2[a:b], p1[b:]])
    c2 = np.concatenate([p2[:a], p1[a:b], p2[b:]])
    return c1, c2
def gaussian_mutation(child, mutation_prob_gene=0.02, sigma=0.1, bounds=(-5.12,5.12)):
    d = child.shape[0]
    mask = np.random.rand(d) < mutation_prob_gene
    noise = np.random.normal(0, sigma, size=d) * mask
    child = child + noise
    child = np.clip(child, bounds[0], bounds[1])
    return child
