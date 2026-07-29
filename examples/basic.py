"""Minimal Bayesian-optimization example."""

from mojo_skopt import gp_minimize


def objective(point):
    x, depth = point
    return (x - 0.25) ** 2 + 0.02 * (depth - 4) ** 2


result = gp_minimize(
    objective,
    [(-1.0, 1.0), (1, 8)],
    n_calls=18,
    n_initial_points=6,
    random_state=7,
)
print(result.x, result.fun)
