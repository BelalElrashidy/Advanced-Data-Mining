"""Minimal PSO minimizer.

This file exposes a compact `PSO` function for minimizing a continuous
objective. It keeps the API minimal: `func`, `bounds`, `num_particles`,
`max_iter`, and `seed`. The function returns (best_position, best_score).

There is also a tiny `__main__` demonstration but no advanced features.
"""

from typing import Callable, Sequence, Tuple, Optional, List
import numpy as np


def PSO(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    num_particles: int = 30,
    max_iter: int = 100,
    seed: Optional[int] = None,
    return_history: bool = False,
) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, List[float]]:
    """Minimal PSO minimizer.

    Parameters
    - func: callable accepting a 1-D numpy array and returning a scalar.
    - bounds: sequence of (low, high) for each dimension.
    - num_particles: swarm size.
    - max_iter: number of iterations.
    - seed: RNG seed.

    Returns (best_position, best_value).
    """

    if seed is not None:
        np.random.seed(seed)

    bounds_arr = np.array(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must be sequence of (low, high) pairs")

    low = bounds_arr[:, 0]
    high = bounds_arr[:, 1]
    dim = len(bounds_arr)

    # simple PSO defaults
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    # initialize
    pos = np.random.uniform(low=low, high=high, size=(num_particles, dim))
    vel = np.zeros_like(pos)

    pbest = pos.copy()
    pbest_scores = np.array([func(p) for p in pbest])

    gbest_idx = int(np.argmin(pbest_scores))
    gbest = pbest[gbest_idx].copy()
    gbest_score = float(pbest_scores[gbest_idx])

    history: List[float] = []

    for _ in range(max_iter):
        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = pos + vel

        # keep in bounds
        pos = np.clip(pos, low, high)

        scores = np.array([func(p) for p in pos])
        improved = scores < pbest_scores
        if np.any(improved):
            pbest[improved] = pos[improved]
            pbest_scores[improved] = scores[improved]

        min_idx = int(np.argmin(pbest_scores))
        if pbest_scores[min_idx] < gbest_score:
            gbest_score = float(pbest_scores[min_idx])
            gbest = pbest[min_idx].copy()

        if return_history:
            history.append(gbest_score)

    if return_history:
        return gbest, gbest_score, history
    return gbest, gbest_score


if __name__ == "__main__":
    # tiny smoke test
    def sphere(x: np.ndarray) -> float:
        return float((x ** 2).sum())

    best_x, best_f = PSO(sphere, bounds=[(-5, 5), (-5, 5)], num_particles=20, max_iter=100, seed=0)
    print(best_x, best_f)
