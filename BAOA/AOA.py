"""Minimal Bat-inspired Optimization (BAOA) implementation.

This is a compact implementation inspired by the Bat Algorithm family. It is
intended for experimentation and follows a simple continuous-minimization API:

    BAOA(func, bounds, num_agents=30, max_iter=100, seed=None, return_history=False)

Returns (best_x, best_f) or (best_x, best_f, history) when return_history=True.
"""

from typing import Callable, Sequence, Tuple, Optional
import numpy as np


def AOA(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    num_agents: int = 30,
    max_iter: int = 100,
    seed: Optional[int] = None,
    return_history: bool = False,
):
    if seed is not None:
        np.random.seed(seed)

    bounds_arr = np.array(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must be sequence of (low, high) pairs")

    low = bounds_arr[:, 0]
    high = bounds_arr[:, 1]
    dim = len(bounds_arr)

    # initialize
    X = np.random.uniform(low=low, high=high, size=(num_agents, dim))
    V = np.zeros_like(X)
    Q = np.zeros(num_agents)  # frequency
    A = 0.9 * np.ones(num_agents)  # loudness
    r = 0.1 * np.ones(num_agents)  # pulse rate

    fitness = np.array([func(x) for x in X])
    best_idx = int(np.argmin(fitness))
    best_x = X[best_idx].copy()
    best_f = float(fitness[best_idx])

    history = []

    Qmin, Qmax = 0.0, 2.0

    for t in range(max_iter):
        for i in range(num_agents):
            Q[i] = Qmin + (Qmax - Qmin) * np.random.rand()
            V[i] = V[i] + (X[i] - best_x) * Q[i]
            S = X[i] + V[i]
            # local random walk
            if np.random.rand() > r[i]:
                S = best_x + 0.001 * np.random.randn(dim)

            S = np.clip(S, low, high)
            fS = func(S)

            # accept with some criteria using loudness
            if (fS <= fitness[i]) and (np.random.rand() < A[i]):
                X[i] = S
                fitness[i] = fS
                A[i] *= 0.9
                r[i] = r[i] * (1 - np.exp(-0.1 * t))

            # update global best
            if fitness[i] < best_f:
                best_f = float(fitness[i])
                best_x = X[i].copy()

        if return_history:
            history.append(best_f)

    if return_history:
        return best_x, best_f, history
    return best_x, best_f


if __name__ == "__main__":
    # tiny smoke test: sphere
    def sphere(x: np.ndarray) -> float:
        return float((x ** 2).sum())

    bx, bf = BAOA(sphere, bounds=[(-5, 5), (-5, 5)], num_agents=20, max_iter=100, seed=1, return_history=False)
    print("BAOA example:", bx, bf)
