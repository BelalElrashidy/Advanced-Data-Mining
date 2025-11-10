# Advanced-Data-Mining — Optimization Examples

Small repository with minimal implementations of two population-based optimizers:

- `PSO.py` — minimal Particle Swarm Optimization (PSO) minimizer
- `SCA.py` — minimal Sine-Cosine Algorithm (SCA) minimizer
- `run_pso_minimal.py` — runner that applies `PSO` to the Ackley function (2-D)
- `run_SCA_minimal.py` — runner that applies `SCA` to the Ackley function (2-D)

Other scripts in the workspace may exist for different experiments.

## Quick start

1. Make sure you have Python 3.8+.

Install the repository dependencies with pip (recommended):

```bash
# install deps (macOS/Linux)
python3 -m pip install --user -r requirements.txt
```

2. Run the PSO example (optimizes Ackley with recommended parameters a=20, b=0.2, c=2\*pi):

```bash
python3 run_pso_minimal.py
```

3. Run the SCA example (same Ackley objective):

```bash
python3 run_SCA_minimal.py
```

Both example runners print the best-found position and objective value. They are intentionally minimal so you can adapt parameters quickly.

## Files and purpose

- `PSO.py` — exposes `PSO(func, bounds, num_particles=30, max_iter=100, seed=None)` and returns `(best_x, best_f)`.
- `SCA.py` — exposes `SCA(func, bounds, num_agents=30, max_iter=100, seed=None)` and returns `(best_x, best_f)`.
- `run_pso_minimal.py` — defines the Ackley function with recommended parameters (a=20, b=0.2, c=2\*pi) and calls `PSO` with a 2-D search space.
- `run_SCA_minimal.py` — same as above but calls `SCA`.

## Ackley function used in runners

The Ackley function used is:

```
-a * exp(-b * sqrt(1/d * sum(x^2))) - exp(1/d * sum(cos(c*x))) + a + e
```

Recommended values used in the runners:

- a = 20
- b = 0.2
- c = 2\*pi

Global minimum is at x = 0 with value 0.

## Getting convergence data and plotting

The current minimal implementations print final results but do not return per-iteration history by default.

If you want convergence curves (best value per iteration), two simple options:

1. Edit the runner (recommended)

   - Change the runner to call the optimizer inside a loop and append the best value each iteration to a Python list. For `PSO` you can collect the global best after each PSO iteration (modify `PSO.py`) or re-run a loop that records it.

2. Modify `PSO.py` and `SCA.py` to add an optional `return_history=False` parameter which returns `(best_x, best_f, history)` where `history` is a list of best values per iteration. This is straightforward to implement (append to a list inside the main loop).

Once you have the histories you can plot them with matplotlib. Example (after you have two lists `hist_pso` and `hist_sca`):

```python
import matplotlib.pyplot as plt

plt.plot(hist_pso, label='PSO')
plt.plot(hist_sca, label='SCA')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Best objective')
plt.legend()
plt.grid(True)
plt.show()
```

## Suggestions / next steps

- Add `return_history=True` to both `PSO.py` and `SCA.py` so runners can plot convergence automatically.
- Add a `plot_convergence.py` script that runs both algorithms and saves a comparison plot (PNG).
- Implement a restart wrapper that runs each optimizer multiple times and returns the best across runs.
- Integrate a local-refinement step (your `hillclimb.py`) to polish solutions from PSO or SCA.

If you'd like, I can implement any of these next steps (add history support, create the plot script, or hook `hillclimb.py` into a hybrid routine) and run a smoke test.

## License and notes

This repository is intended for education and experimentation. The implementations are minimal and not production-tuned.
