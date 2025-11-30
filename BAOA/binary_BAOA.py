"""Binary variant of a BAOA-like optimizer for feature selection.

This implementation keeps continuous internal positions but maps them to
binary masks via a transfer function (sigmoid). The objective function
`func_binary` must accept a 1-D binary array (0/1) and return a scalar to
minimize (lower is better). The algorithm returns the best binary mask and
its objective value; if `return_history=True` it also returns a list of best
objective values per iteration.
"""

from typing import Callable, Sequence, Tuple, Optional, List
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_BAOA(
    func_binary: Callable[[np.ndarray], float],
    n_bits: int,
    num_agents: int = 30,
    max_iter: int = 100,
    seed: Optional[int] = None,
    transfer: str = 'sigmoid',
    return_history: bool = False,
) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, List[float]]:
    """Run a binary BAOA-like optimizer.

    func_binary: callable(mask: np.ndarray) -> float (minimize)
    n_bits: number of bits (features)
    transfer: currently only 'sigmoid' supported
    """
    if seed is not None:
        np.random.seed(seed)

    # continuous positions in R (initialized near 0)
    X = np.random.uniform(low=-1.0, high=1.0, size=(num_agents, n_bits))
    V = np.zeros_like(X)
    Q = np.zeros(num_agents)
    A = 0.9 * np.ones(num_agents)
    r = 0.1 * np.ones(num_agents)

    # map initial to binaries and evaluate
    if transfer == 'sigmoid':
        P = _sigmoid(X)
    else:
        P = _sigmoid(X)
    B = (np.random.rand(*P.shape) < P).astype(int)
    fitness = np.array([func_binary(b) for b in B])

    best_idx = int(np.argmin(fitness))
    best_b = B[best_idx].copy()
    best_f = float(fitness[best_idx])

    history: List[float] = []

    Qmin, Qmax = 0.0, 2.0

    for t in range(max_iter):
        for i in range(num_agents):
            Q[i] = Qmin + (Qmax - Qmin) * np.random.rand()
            V[i] = V[i] + (X[i] - _sigmoid(best_b)) * Q[i]
            S = X[i] + V[i]
            # local random walk
            if np.random.rand() > r[i]:
                S = _sigmoid(best_b) + 0.001 * np.random.randn(n_bits)

            # map S -> probabilities
            if transfer == 'sigmoid':
                Pnew = _sigmoid(S)
            else:
                Pnew = _sigmoid(S)

            Bnew = (np.random.rand(n_bits) < Pnew).astype(int)
            fnew = func_binary(Bnew)

            # accept with loudness criteria
            if (fnew <= fitness[i]) and (np.random.rand() < A[i]):
                X[i] = S
                B[i] = Bnew
                fitness[i] = fnew
                A[i] *= 0.9
                r[i] = r[i] * (1 - np.exp(-0.1 * t)) if t > 0 else r[i]

            # update global best
            if fitness[i] < best_f:
                best_f = float(fitness[i])
                best_b = B[i].copy()

        if return_history:
            history.append(best_f)

    if return_history:
        return best_b, best_f, history
    return best_b, best_f


if __name__ == "__main__":
    # tiny smoke test: optimize a toy objective that counts selected bits
    def obj(mask: np.ndarray) -> float:
        # prefer half the bits set (toy)
        return abs(mask.sum() - (mask.size // 2))

    b, f = binary_BAOA(obj, n_bits=10, num_agents=10, max_iter=30, seed=1, return_history=False)
    print('best mask', b, 'f', f)
