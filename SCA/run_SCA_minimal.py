"""Runner for the minimal SCA implementation using the Ackley function.
"""

from typing import List, Tuple
import numpy as np
from SCA.SCA import SCA


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_sq = np.sum(x ** 2)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return float(term1 + term2 + a + np.e)


def main():
    bounds: List[Tuple[float, float]] = [(-5.0, 5.0), (-5.0, 5.0)]
    for i in [50, 100, 500]:
        print(f"\nRunning SCA with max_iter={i}")
        best_x, best_f = SCA(lambda x: ackley(x, a=20.0, b=0.2, c=2 * np.pi), bounds, num_agents=40, max_iter=i, seed=42)
        print("SCA best_x:", best_x)
        print("SCA best_f:", best_f)


if __name__ == "__main__":
    main()
