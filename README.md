# silicon-natural-gradient-test

Minimal prototype testing Fisher-Rao natural-gradient flow vs Euclidean updates on Gaussian sampling distributions.

This is the silicon-systems instantiation of the information-geometric selection framework (natural-gradient ascent of a performance functional on the Fisher–Rao manifold = replicator dynamics).

## Run

```bash
python silicon_natural_gradient_test.py
```

Requires numpy. matplotlib is optional (plots are skipped if absent).

## What it measures

- Mean cost on sphere / ellipsoid / Rosenbrock
- KL divergence from the initial distribution
- Fitness variance (predicted squared speed under the Fisher metric)

Natural-gradient steps use the closed-form Fisher metric for a diagonal Gaussian; Euclidean steps do not.
