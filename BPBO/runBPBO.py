import numpy as np
from BPBO.BPBO import bpbo
import matplotlib.pyplot as plt

def rastrigin(x, A=10):
    x = np.asarray(x)
    n = x.size
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def obj_max_neg_rastrigin(x):
    # we maximize -Rastrigin so optimizer seeks 0 vector (global optimum)
    return rastrigin(x)

# run PBPO for 5 independent restarts and collect results
results = []
print("Pop Size Changing")
# for run in range(5):
prey, prey_fitness, hist = bpbo(obj_max_neg_rastrigin,
         dim=2,
         bounds=(-5.12,5.12),
         pop_size=10,
         max_iter=500,
         seed=42,
         verbose=False)
results.append((prey, prey_fitness,hist))
    
prey, prey_fitness, hist = bpbo(obj_max_neg_rastrigin,
         dim=2,
         bounds=(-5.12,5.12),
         pop_size=30,
         max_iter=500,
         seed=42,
         verbose=False)
results.append((prey, prey_fitness,hist))
    
prey, prey_fitness, hist = bpbo(obj_max_neg_rastrigin,
    dim=2,
    bounds=(-5.12,5.12),
    pop_size=70,
    max_iter=500,
    seed=42,
    verbose=False)
results.append((prey, prey_fitness,hist))

for i in range(3):
    
    print(f"Run {i+1}: Prey_fit={results[i][1]:.6f}, Prey={results[i][0]}")

# compute mean and std of the best evaluations
best_vals = np.array([r[1] for r in results])
print("\nSummary of runs:")
print("mean:", best_vals.mean())
print("std: ", best_vals.std())

