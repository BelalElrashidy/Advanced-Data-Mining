"""Parameter study: vary number of iterations and compare optimizers.

This script runs each algorithm (PSO, SCA, BAOA) for several values of
`max_iter`, repeats each experiment a few times with different seeds, and
plots the mean and standard deviation of the final best objective vs
iteration count.

Output: `param_iterations.png` in repository root.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PSO.PSO import PSO
from SCA.SCA import SCA
from BAOA.AOA import BAOA


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_sq = np.sum(x ** 2)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return float(term1 + term2 + a + np.e)


def run_once(method: str, max_iter: int, seed: int, num_agents: int, bounds):
    if method == 'pso':
        _, best_f = PSO(ackley, bounds, num_particles=num_agents, max_iter=max_iter, seed=seed)
    elif method == 'sca':
        _, best_f = SCA(ackley, bounds, num_agents=num_agents, max_iter=max_iter, seed=seed)
    elif method == 'baoa':
        _, best_f = BAOA(ackley, bounds, num_agents=num_agents, max_iter=max_iter, seed=seed)
    else:
        raise ValueError(method)
    return best_f


def main():
    parser = argparse.ArgumentParser(description='Parameter study: iterations vs performance')
    parser.add_argument('--methods', nargs='+', default=['pso', 'sca', 'baoa'], help='methods to run')
    parser.add_argument('--iters', nargs='+', type=int, default=[50, 100, 200, 500], help='iteration counts')
    parser.add_argument('--runs', type=int, default=5, help='repeats per config (for mean/std)')
    parser.add_argument('--agents', type=int, default=40, help='number of particles/agents')
    parser.add_argument('--out', default='param_iterations.png', help='output image filename')
    args = parser.parse_args()

    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    methods = args.methods
    iters = args.iters
    runs = args.runs
    agents = args.agents

    results = {m: [] for m in methods}

    for m in methods:
        print(f'Running method {m}')
        for it in iters:
            vals = []
            for r in range(runs):
                seed = 1000 + r
                best_f = run_once(m, it, seed, agents, bounds)
                vals.append(best_f)
            vals = np.array(vals)
            mean = vals.mean()
            std = vals.std()
            results[m].append((mean, std))
            print(f'  it={it} mean={mean:.4e} std={std:.4e}')

    # Plot
    plt.figure(figsize=(8, 5))
    for m in methods:
        means = [x[0] for x in results[m]]
        stds = [x[1] for x in results[m]]
        plt.errorbar(iters, means, yerr=stds, label=m.upper(), marker='o')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Final best objective (mean ± std, log scale)')
    plt.title('Iteration budget vs final performance')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print('Saved', args.out)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()
