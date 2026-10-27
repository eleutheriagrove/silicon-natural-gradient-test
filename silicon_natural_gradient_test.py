#!/usr/bin/env python3
"""
Minimal silicon-systems test of natural-gradient (Fisher–Rao) flow
vs Euclidean gradient descent on a Gaussian sampling distribution.

Implements the concrete next steps:
- Gaussian parameter distribution (mean + diagonal covariance for simplicity)
- Performance functional on standard test functions
- Explicit natural-gradient update vs Euclidean update
- Logging of KL from initial measure, fitness variance, and convergence
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available; skipping plots")

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def sphere(x):
    return np.sum(x**2)

def ellipsoid(x):
    n = len(x)
    return np.sum((10**(6 * np.arange(n) / (n - 1))) * x**2)

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

FUNCTIONS = {
    "sphere": sphere,
    "ellipsoid": ellipsoid,
    "rosenbrock": rosenbrock,
}

# ---------------------------------------------------------------------------
# Gaussian natural-gradient utilities (diagonal covariance for tractability)
# ---------------------------------------------------------------------------

def sample_gaussian(mean, log_std, n_samples):
    std = np.exp(log_std)
    return mean + std * np.random.randn(n_samples, len(mean))

def log_prob_gaussian(x, mean, log_std):
    std = np.exp(log_std)
    return -0.5 * np.sum(((x - mean) / std)**2 + 2 * log_std + np.log(2 * np.pi), axis=-1)

def kl_from_initial(mean, log_std, mean0, log_std0):
    """KL(p || p0) for diagonal Gaussians (closed form)."""
    var = np.exp(2 * log_std)
    var0 = np.exp(2 * log_std0)
    return 0.5 * np.sum(
        (var + (mean - mean0)**2) / var0 - 1 + 2 * (log_std0 - log_std)
    )

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def natural_gradient_step(mean, log_std, samples, fitnesses, eta=0.1):
    """
    Natural-gradient ascent on expected fitness for diagonal Gaussian.
    Fisher metric for (mean, log_std) is diagonal and known in closed form.
    """
    n, d = samples.shape
    # Weighted scores (natural gradient of expected fitness)
    weights = fitnesses - np.mean(fitnesses)  # center for variance interpretation
    weights = weights / (np.std(weights) + 1e-8)

    # Score for mean: (x - mean) / var
    std = np.exp(log_std)
    score_mean = (samples - mean) / (std**2)
    # Score for log_std: ((x - mean)/std)**2 - 1
    score_logstd = ((samples - mean) / std)**2 - 1

    grad_mean = np.mean(weights[:, None] * score_mean, axis=0)
    grad_logstd = np.mean(weights[:, None] * score_logstd, axis=0)

    # Fisher is identity (after parameterization) for this choice of coordinates
    # so natural gradient ≈ ordinary gradient of expected fitness
    mean_new = mean + eta * grad_mean
    log_std_new = log_std + eta * grad_logstd
    return mean_new, log_std_new

def euclidean_gradient_step(mean, log_std, samples, fitnesses, eta=0.01):
    """Plain Euclidean gradient on the parameters (no Fisher correction)."""
    n, d = samples.shape
    weights = fitnesses - np.mean(fitnesses)
    weights = weights / (np.std(weights) + 1e-8)

    # Finite-difference style Euclidean gradient w.r.t. mean and log_std
    grad_mean = np.mean(weights[:, None] * (samples - mean), axis=0)
    std = np.exp(log_std)
    grad_logstd = np.mean(weights[:, None] * ((samples - mean)**2 / std - std), axis=0)

    mean_new = mean + eta * grad_mean
    log_std_new = log_std + eta * 0.1 * grad_logstd  # smaller step on variance
    return mean_new, log_std_new

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def run_comparison(func_name="sphere", dim=10, n_samples=32, n_steps=100, seed=0):
    np.random.seed(seed)
    f = FUNCTIONS[func_name]

    # Initial distribution
    mean0 = np.zeros(dim)
    log_std0 = np.zeros(dim)

    results = {}
    for method, step_fn, eta in [
        ("natural", natural_gradient_step, 0.2),
        ("euclidean", euclidean_gradient_step, 0.05),
    ]:
        mean = mean0.copy()
        log_std = log_std0.copy()
        history = {
            "mean_fitness": [],
            "kl": [],
            "fitness_var": [],
            "best": [],
        }
        for t in range(n_steps):
            samples = sample_gaussian(mean, log_std, n_samples)
            fitnesses = -np.array([f(s) for s in samples])  # maximize negative cost
            history["mean_fitness"].append(np.mean(-fitnesses))  # report positive cost
            history["kl"].append(kl_from_initial(mean, log_std, mean0, log_std0))
            history["fitness_var"].append(np.var(fitnesses))
            history["best"].append(np.min(-fitnesses))
            mean, log_std = step_fn(mean, log_std, samples, fitnesses, eta=eta)
        results[method] = history
    return results

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, func_name):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method, hist in results.items():
        axes[0].plot(hist["mean_fitness"], label=method)
        axes[1].plot(hist["kl"], label=method)
        axes[2].plot(hist["fitness_var"], label=method)
    axes[0].set_title(f"{func_name}: mean cost")
    axes[1].set_title("KL from initial")
    axes[2].set_title("fitness variance")
    for ax in axes:
        ax.legend()
        ax.set_xlabel("step")
    plt.tight_layout()
    plt.savefig(f"silicon_natural_gradient_{func_name}.png", dpi=150)
    plt.close()
    print(f"Saved plot: silicon_natural_gradient_{func_name}.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for fn in ["sphere", "ellipsoid", "rosenbrock"]:
        print(f"\n=== {fn} ===")
        res = run_comparison(fn, dim=10, n_samples=64, n_steps=80)
        for method, hist in res.items():
            print(f"  {method:10s}  final mean cost: {hist['mean_fitness'][-1]:.4f}  "
                  f"final KL: {hist['kl'][-1]:.4f}  "
                  f"avg var: {np.mean(hist['fitness_var']):.4f}")
        plot_results(res, fn)
    print("\nDone.")
