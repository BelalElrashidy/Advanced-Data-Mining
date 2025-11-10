import numpy as np
def objective(x,A=20):
    n = len(x)  # number of dimensions
    return -(A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x]))


def generate_neighbors(x, step_size=0.1):
    neighbors = []
    for i in range(len(x)):
        step = np.zeros_like(x)
        step[i] = step_size
        neighbors.append(x + step)
        neighbors.append(x - step)
    return neighbors


def hill_climbing(objective, initial, n_iterations=100, step_size=0.1):
    current = np.array([initial])
    current_eval = objective(current)
    for i in range(n_iterations):
        neighbors = generate_neighbors(current, step_size)
        neighbor_evals = [objective(n) for n in neighbors]

        best_idx = np.argmax(neighbor_evals)
        if neighbor_evals[best_idx] > current_eval:
            current = neighbors[best_idx]
            current_eval = neighbor_evals[best_idx]
            print(
                f"Step {i+1}: x = {current[0]:.4f}, f(x) = {current_eval:.4f}")
        else:
            print("No better neighbors found. Algorithm converged.")
            break
    return current, current_eval


initial_guess = 10.0
solution, value = hill_climbing(
    objective, initial_guess, n_iterations=50, step_size=0.0001)
print(f"\nBest solution x = {solution[0]:.4f}, f(x) = {value:.4f}")