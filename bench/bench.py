"""Benchmarks against scikit-optimize on identical inputs."""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from mojo_skopt import acquisition as mojo_acquisition
from mojo_skopt._lib import addr, f64, lib
from mojo_skopt.learning import (
    ExtraTreesRegressor as MojoExtraTreesRegressor,
    GaussianProcessRegressor as MojoGaussianProcessRegressor,
)
from skopt import acquisition as skopt_acquisition
from skopt.learning import (
    ExtraTreesRegressor as SkoptExtraTreesRegressor,
    GaussianProcessRegressor as SkoptGaussianProcessRegressor,
)


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


class StaticPosterior:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def predict(self, X, return_std=False):
        return (self.mean, self.std) if return_std else self.mean


def covariance_case():
    rng = np.random.RandomState(0)
    X = f64(rng.normal(size=(3000, 8)))
    Y = f64(rng.normal(size=(500, 8)))
    scale = f64(np.linspace(0.4, 1.2, 8))
    destination = np.empty((len(X), len(Y)))
    kernel = Matern(scale, nu=2.5)

    def ours():
        lib().msko_covariance(
            addr(X),
            addr(Y),
            addr(destination),
            len(X),
            len(Y),
            X.shape[1],
            addr(scale),
            1.0,
            3,
        )
        return destination

    return ours, lambda: kernel(X, Y)


def acquisition_case():
    rng = np.random.RandomState(1)
    mean = f64(rng.normal(size=2_000_000))
    std = f64(rng.uniform(0.01, 2.0, size=len(mean)))
    X = np.empty((len(mean), 0))
    posterior = StaticPosterior(mean, std)
    return (
        lambda: mojo_acquisition.gaussian_ei(X, posterior, y_opt=-1.0, xi=0.01),
        lambda: skopt_acquisition.gaussian_ei(X, posterior, y_opt=-1.0, xi=0.01),
    )


def gp_fit_case():
    rng = np.random.RandomState(2)
    X = rng.uniform(-1, 1, (650, 6))
    y = np.sin(4 * X[:, 0]) + X[:, 1] - X[:, 2] ** 2
    kernel = ConstantKernel(1.0) * Matern(np.full(6, 0.7), nu=2.5)
    kwargs = dict(kernel=kernel, alpha=1e-7, optimizer=None, normalize_y=True)
    return (
        lambda: MojoGaussianProcessRegressor(**kwargs).fit(X, y),
        lambda: SkoptGaussianProcessRegressor(**kwargs).fit(X, y),
    )


def gp_predict_case():
    rng = np.random.RandomState(3)
    X = rng.uniform(-1, 1, (160, 8))
    y = np.sin(4 * X[:, 0]) + X[:, 1]
    Q = rng.uniform(-1, 1, (20_000, 8))
    kernel = ConstantKernel(1.0) * Matern(np.full(8, 0.6), nu=2.5)
    kwargs = dict(kernel=kernel, alpha=1e-7, optimizer=None, normalize_y=True)
    ours = MojoGaussianProcessRegressor(**kwargs).fit(X, y)
    theirs = SkoptGaussianProcessRegressor(**kwargs).fit(X, y)
    return lambda: ours.predict(Q, return_std=True), lambda: theirs.predict(Q, return_std=True)


def forest_predict_case():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(20_000, 10))
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
    Q = rng.normal(size=(100_000, 10))
    kwargs = dict(
        n_estimators=100,
        min_samples_leaf=2,
        min_variance=1e-8,
        random_state=5,
        n_jobs=1,
    )
    ours = MojoExtraTreesRegressor(**kwargs).fit(X, y)
    theirs = SkoptExtraTreesRegressor(**kwargs).fit(X, y)
    return lambda: ours.predict(Q, return_std=True), lambda: theirs.predict(Q, return_std=True)


CASES = [
    ("Matern covariance (3000 x 500 x 8)", covariance_case),
    ("Expected improvement (2M)", acquisition_case),
    ("GP.fit (650 x 6)", gp_fit_case),
    ("GP.predict mean+std (160 train, 20k query)", gp_predict_case),
    ("ExtraTrees.predict mean+std (100 trees, 100k)", forest_predict_case),
]


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| case | mojo-scikit-optimize | scikit-optimize | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, builder in CASES:
        ours, theirs = builder()
        actual = ours()
        expected = theirs()
        if hasattr(actual, "alpha_"):
            assert np.allclose(actual.alpha_, expected.alpha_, rtol=2e-7, atol=1e-7)
        elif isinstance(actual, tuple):
            assert all(np.allclose(a, b, rtol=2e-7, atol=1e-7) for a, b in zip(actual, expected))
        else:
            assert np.allclose(actual, expected, rtol=2e-7, atol=1e-7)
        mojo_seconds = timeit(ours)
        skopt_seconds = timeit(theirs)
        ratio = skopt_seconds / mojo_seconds
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{skopt_seconds * 1000:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
