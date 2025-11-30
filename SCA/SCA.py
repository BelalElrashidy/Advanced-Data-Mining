"""Minimal Sine-Cosine Algorithm (SCA) implementation.

Exposes SCA(func, bounds, num_agents=30, max_iter=100, seed=None) -> (best_x, best_f)
"""

from typing import Callable, Sequence, Tuple, Optional, List
import numpy as np


def SCA(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    num_agents: int = 30,
    max_iter: int = 100,
    seed: Optional[int] = None,
    return_history: bool = False,
) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, List[float]]:
    """Run a minimal Sine-Cosine Algorithm to minimize `func`.

    Parameters
    - func: takes a 1-D numpy array and returns a scalar.
    - bounds: sequence of (low, high) pairs per dimension.
    - num_agents: number of candidate solutions.
    - max_iter: number of iterations.
    - seed: RNG seed.

    Returns (best_position, best_value)
    """
    if seed is not None:
        np.random.seed(seed)

    bounds_arr = np.array(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must be sequence of (low, high) pairs")

    low = bounds_arr[:, 0]
    high = bounds_arr[:, 1]
    dim = len(bounds_arr)

    # initialize agents
    X = np.random.uniform(low=low, high=high, size=(num_agents, dim))
    fitness = np.array([func(x) for x in X])
    best_idx = int(np.argmin(fitness))
    best_x = X[best_idx].copy()
    best_f = float(fitness[best_idx])

    history: List[float] = []

    # SCA loop
    for t in range(max_iter):
        r1 = 2 - t * (2 / max_iter)  # linearly decreasing from 2 to 0
        for i in range(num_agents):
            r2 = 2 * np.pi * np.random.rand(dim)
            r3 = 2 * np.random.rand(dim)
            r4 = np.random.rand(dim)

            # update position
            cond = r4 < 0.5
            # sine move
            X[i] = np.where(
                cond,
                X[i] + r1 * np.sin(r2) * np.abs(r3 * best_x - X[i]),
                X[i] + r1 * np.cos(r2) * np.abs(r3 * best_x - X[i]),
            )

            # keep in bounds
            X[i] = np.clip(X[i], low, high)

        fitness = np.array([func(x) for x in X])
        min_idx = int(np.argmin(fitness))
        if fitness[min_idx] < best_f:
            best_f = float(fitness[min_idx])
            best_x = X[min_idx].copy()

        if return_history:
            history.append(best_f)

    if return_history:
        return best_x, best_f, history
    return best_x, best_f

