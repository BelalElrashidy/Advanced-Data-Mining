"""Run minimal PSO on the Ackley function (user-specified params).

Ackley function:
  -a * exp(-b * sqrt(1/d * sum(x^2))) - exp(1/d * sum(cos(c*x))) + a + e

Recommended parameters: a=20, b=0.2, c=2*pi
"""

from typing import List, Tuple
import numpy as np
from PSO.PSO import PSO


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_sq = np.sum(x ** 2)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return float(term1 + term2 + a + math_e)


math_e = np.e


def main():
    # 2-D Ackley with bounds [-5, 5]
    bounds: List[Tuple[float, float]] = [(-5.0, 5.0), (-5.0, 5.0)]
    for i in [50, 100, 500]:
        print(f"\nRunning PSO with max_iter={i}")
        best_x, best_f = PSO(lambda x: ackley(x, a=20.0, b=0.2, c=2 * np.pi), bounds, num_particles=40, max_iter=i, seed=42)
        print("Best position:", best_x)
        print("Best value:", best_f)
    


if __name__ == "__main__":
    main()
"""Runner for the minimal PSO implementation.

This script imports `PSO` from `PSO.py`, defines a simple objective and prints
the result. It's intended as a minimal example that can be executed directly.
"""

# from PSO import PSO
# import numpy as np


# def sphere(x: np.ndarray) -> float:
#     return float((x ** 2).sum())


# def main():
#     best_x, best_f = PSO(sphere, bounds=[(-10, 10), (-10, 10)], num_particles=30, max_iter=150, seed=42)
#     print("best_x:", best_x)
#     print("best_f:", best_f)


# if __name__ == "__main__":
#     main()
