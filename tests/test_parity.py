"""Numerical and behavioural parity with scikit-optimize 0.10."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

import mojo_skopt as mskopt
from mojo_skopt import acquisition as macq
from mojo_skopt._lib import addr, f64, lib
from mojo_skopt.learning import (
    ExtraTreesRegressor,
    GaussianProcessRegressor,
    RandomForestRegressor,
)
from mojo_skopt.space import Categorical, Integer, Real, Space
from mojo_skopt.utils import use_named_args

skopt = pytest.importorskip("skopt")
from skopt import acquisition as sacq
from skopt.learning import (
    ExtraTreesRegressor as SkExtraTreesRegressor,
    GaussianProcessRegressor as SkGaussianProcessRegressor,
    RandomForestRegressor as SkRandomForestRegressor,
)
from skopt.learning.gaussian_process import kernels as sk_kernels


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.RandomState(4)
    X = rng.uniform(-1, 1, (45, 4))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2]
    Q = rng.uniform(-1, 1, (31, 4))
    return X, y, Q


def test_shared_library_is_the_requested_artifact():
    expected = Path(__file__).resolve().parents[1] / "dist" / "libmojo-scikit-optimize.so"
    assert expected.is_file()
    assert expected.stat().st_size > 10_000


@pytest.mark.parametrize(
    ("kernel", "kind"),
    [
        (RBF([0.3, 0.7, 1.1]), 0),
        (Matern([0.3, 0.7, 1.1], nu=0.5), 1),
        (Matern([0.3, 0.7, 1.1], nu=1.5), 2),
        (Matern([0.3, 0.7, 1.1], nu=2.5), 3),
    ],
)
def test_covariance_kernel_parity(kernel, kind):
    rng = np.random.RandomState(0)
    X = f64(rng.normal(size=(17, 3)))
    Y = f64(rng.normal(size=(13, 3)))
    scale = f64([0.3, 0.7, 1.1])
    result = np.empty((len(X), len(Y)))
    lib().msko_covariance(
        addr(X), addr(Y), addr(result), len(X), len(Y), 3, addr(scale), 1.7, kind
    )
    assert np.allclose(result, 1.7 * kernel(X, Y), rtol=2e-11, atol=2e-13)


def test_simd_cholesky_and_solve_tail():
    rng = np.random.RandomState(12)
    raw = rng.normal(size=(13, 13))
    matrix = f64(raw @ raw.T + np.eye(13) * 0.25)
    expected_factor = np.linalg.cholesky(matrix)
    factor = matrix.copy()
    assert lib().msko_cholesky(addr(factor), len(factor)) == 1
    assert np.allclose(factor, expected_factor, rtol=2e-13, atol=2e-13)

    rhs = f64(rng.normal(size=13), copy=True)
    expected_solution = np.linalg.solve(matrix, rhs)
    lib().msko_cholesky_solve(addr(factor), addr(rhs), len(rhs))
    assert np.allclose(rhs, expected_solution, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "kernel",
    [
        ConstantKernel(1.3) * RBF([0.2, 0.5, 0.9, 1.2]),
        ConstantKernel(0.8) * Matern([0.2, 0.5, 0.9, 1.2], nu=0.5),
        ConstantKernel(0.8) * Matern([0.2, 0.5, 0.9, 1.2], nu=1.5),
        ConstantKernel(0.8) * Matern([0.2, 0.5, 0.9, 1.2], nu=2.5),
    ],
)
def test_gaussian_process_prediction_parity(regression_data, kernel):
    X, y, Q = regression_data
    ours = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, alpha=1e-6, normalize_y=True
    ).fit(X, y)
    theirs = SkGaussianProcessRegressor(
        kernel=kernel, optimizer=None, alpha=1e-6, normalize_y=True
    ).fit(X, y)
    mean, std = ours.predict(Q, return_std=True)
    expected_mean, expected_std = theirs.predict(Q, return_std=True)
    assert np.allclose(mean, expected_mean, rtol=2e-10, atol=2e-10)
    assert np.allclose(std, expected_std, rtol=2e-10, atol=2e-10)
    mean_cov, covariance = ours.predict(Q[:7], return_cov=True)
    expected_mean_cov, expected_covariance = theirs.predict(Q[:7], return_cov=True)
    assert np.allclose(mean_cov, expected_mean_cov, atol=2e-10)
    assert np.allclose(covariance, expected_covariance, atol=2e-10)


@pytest.mark.parametrize(("our_kernel", "their_kernel"), [
    (
        ConstantKernel(1.1) * RBF([0.4, 0.6, 0.8, 1.0]),
        sk_kernels.ConstantKernel(1.1) * sk_kernels.RBF([0.4, 0.6, 0.8, 1.0]),
    ),
    (
        ConstantKernel(1.1) * Matern([0.4, 0.6, 0.8, 1.0], nu=2.5),
        sk_kernels.ConstantKernel(1.1)
        * sk_kernels.Matern([0.4, 0.6, 0.8, 1.0], nu=2.5),
    ),
])
def test_gaussian_process_gradient_parity(regression_data, our_kernel, their_kernel):
    X, y, Q = regression_data
    ours = GaussianProcessRegressor(
        kernel=our_kernel, optimizer=None, alpha=1e-6, normalize_y=True
    ).fit(X, y)
    theirs = SkGaussianProcessRegressor(
        kernel=their_kernel, optimizer=None, alpha=1e-6, normalize_y=True
    ).fit(X, y)
    actual = ours.predict(
        Q[:1], return_std=True, return_mean_grad=True, return_std_grad=True
    )
    expected = theirs.predict(
        Q[:1], return_std=True, return_mean_grad=True, return_std_grad=True
    )
    for actual_item, expected_item in zip(actual, expected):
        assert np.allclose(actual_item, expected_item, rtol=2e-8, atol=2e-8)


class _Posterior:
    def __init__(self):
        self.mu = np.array([-1.0, -0.2, 0.5, 1.7])
        self.std = np.array([0.0, 0.1, 0.8, 2.0])

    def predict(
        self, X, return_std=False, return_mean_grad=False, return_std_grad=False
    ):
        n = len(X)
        mu, std = self.mu[:n], self.std[:n]
        if return_mean_grad:
            mu_grad = np.array([0.2, -0.4])
            std_grad = np.array([-0.3, 0.1])
            return mu, std, mu_grad, std_grad
        return (mu, std) if return_std else mu


@pytest.mark.parametrize(
    ("ours", "theirs", "kwargs"),
    [
        (macq.gaussian_lcb, sacq.gaussian_lcb, {"kappa": 2.4}),
        (macq.gaussian_pi, sacq.gaussian_pi, {"y_opt": -0.4, "xi": 0.03}),
        (macq.gaussian_ei, sacq.gaussian_ei, {"y_opt": -0.4, "xi": 0.03}),
    ],
)
def test_acquisition_batch_parity(ours, theirs, kwargs):
    X = np.zeros((4, 2))
    assert np.allclose(ours(X, _Posterior(), **kwargs), theirs(X, _Posterior(), **kwargs))


def test_acquisition_rejects_mismatched_buffers_and_handles_empty_buffers():
    with pytest.raises(ValueError, match="same length"):
        macq._values(np.ones(2), np.ones(1), 2)
    assert macq._values(np.empty(0), np.empty(0), 2).shape == (0,)


def test_ffi_address_rejects_unsafe_buffers():
    with pytest.raises(ValueError, match="empty"):
        addr(np.empty(0, dtype=np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((3, 3), dtype=np.float64)[:, ::2])
    with pytest.raises(TypeError, match="dtype"):
        addr(np.ones(3, dtype=np.int32))
    with pytest.raises(TypeError, match="complex"):
        f64(np.ones(3, dtype=np.complex128))


@pytest.mark.parametrize(
    ("ours", "theirs", "kwargs"),
    [
        (macq.gaussian_lcb, sacq.gaussian_lcb, {"kappa": 2.4}),
        (macq.gaussian_pi, sacq.gaussian_pi, {"y_opt": -0.4, "xi": 0.03}),
        (macq.gaussian_ei, sacq.gaussian_ei, {"y_opt": -0.4, "xi": 0.03}),
    ],
)
def test_acquisition_gradient_parity(ours, theirs, kwargs):
    posterior = _Posterior()
    posterior.mu = posterior.mu[1:2]
    posterior.std = posterior.std[1:2]
    actual = ours(np.zeros((1, 2)), posterior, return_grad=True, **kwargs)
    expected = theirs(np.zeros((1, 2)), posterior, return_grad=True, **kwargs)
    assert np.allclose(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1])


@pytest.mark.parametrize(
    ("ours_class", "theirs_class"),
    [
        (ExtraTreesRegressor, SkExtraTreesRegressor),
        (RandomForestRegressor, SkRandomForestRegressor),
    ],
)
def test_forest_mean_and_conditional_std_parity(regression_data, ours_class, theirs_class):
    X, y, Q = regression_data
    kwargs = dict(n_estimators=31, random_state=7, min_variance=0.02)
    ours = ours_class(**kwargs).fit(X, y)
    theirs = theirs_class(**kwargs).fit(X, y)
    mean, std = ours.predict(Q, return_std=True)
    expected_mean, expected_std = theirs.predict(Q, return_std=True)
    assert np.allclose(mean, expected_mean, rtol=1e-14, atol=1e-14)
    assert np.allclose(std, expected_std, rtol=1e-14, atol=1e-14)


def test_forest_serial_and_parallel_threshold_parity(regression_data):
    X, y, Q = regression_data
    kwargs = dict(n_estimators=11, random_state=13, min_variance=0.02)
    ours = ExtraTreesRegressor(**kwargs).fit(X, y)
    theirs = SkExtraTreesRegressor(**kwargs).fit(X, y)
    for size in (1023, 1024, 1089):
        query = np.resize(Q, (size, Q.shape[1]))
        mean, std = ours.predict(query, return_std=True)
        expected_mean, expected_std = theirs.predict(query, return_std=True)
        assert np.allclose(mean, expected_mean, rtol=1e-14, atol=1e-14)
        assert np.allclose(std, expected_std, rtol=1e-14, atol=1e-14)


def test_estimators_validate_empty_nonfinite_and_unsupported_inputs(regression_data):
    X, y, Q = regression_data
    gp = GaussianProcessRegressor(
        kernel=RBF(1.0), optimizer=None, alpha=np.full(len(X), 1e-6)
    ).fit(X, y)
    assert gp.predict(np.empty((0, X.shape[1]))).shape == (0,)
    with pytest.raises(ValueError, match="finite"):
        gp.fit(X.copy(), np.where(np.arange(len(y)) == 0, np.nan, y))
    with pytest.raises(ValueError, match="nu"):
        GaussianProcessRegressor(kernel=Matern(nu=3.5), optimizer=None).fit(X, y)

    forest = ExtraTreesRegressor(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="0 sample"):
        forest.predict(np.empty((0, X.shape[1])))
    for invalid_value in (np.inf, np.nan):
        invalid = Q.copy()
        invalid[0, 0] = invalid_value
        with pytest.raises(ValueError):
            forest.predict(invalid)


def test_space_roundtrip_and_domain_rules():
    space = Space(
        [
            Real(1e-4, 1e2, prior="log-uniform", name="rate"),
            Integer(2, 8, name="depth"),
            Categorical(["a", "b", "c"], name="method"),
        ]
    )
    points = space.rvs(50, random_state=3)
    roundtripped = space.inverse_transform(space.transform(points))
    assert np.allclose(
        [point[0] for point in roundtripped], [point[0] for point in points], rtol=2e-15
    )
    assert [point[1:] for point in roundtripped] == [point[1:] for point in points]
    assert all(1e-4 <= point[0] <= 1e2 for point in points)
    assert all(2 <= point[1] <= 8 for point in points)
    assert {point[2] for point in points} <= {"a", "b", "c"}


def test_upstream_signatures_for_covered_public_api():
    assert inspect.signature(mskopt.gp_minimize) == inspect.signature(skopt.gp_minimize)
    assert inspect.signature(mskopt.forest_minimize) == inspect.signature(skopt.forest_minimize)
    assert inspect.signature(mskopt.dummy_minimize) == inspect.signature(skopt.dummy_minimize)
    assert inspect.signature(mskopt.Optimizer) == inspect.signature(skopt.Optimizer)
    assert inspect.signature(macq.gaussian_ei) == inspect.signature(sacq.gaussian_ei)


def test_optimizer_ask_tell_and_batch_contract():
    optimizer = mskopt.Optimizer([(-1.0, 1.0)], n_initial_points=3, random_state=2)
    batch = optimizer.ask(n_points=3)
    assert len(batch) == 3 and len({repr(point) for point in batch}) == 3
    optimizer.tell(batch, [(point[0] - 0.15) ** 2 for point in batch])
    point = optimizer.ask()
    result = optimizer.tell(point, (point[0] - 0.15) ** 2)
    assert len(result.x_iters) == 4
    assert result.fun == pytest.approx(min(result.func_vals))
    assert len(result.models) == 2
    duplicate = optimizer.copy(random_state=12)
    assert duplicate.get_result().x_iters == result.x_iters
    assert duplicate.ask() != optimizer.ask()


def test_dummy_minimize_runs_only_requested_objective_calls():
    calls = []

    def objective(point):
        calls.append(point)
        return point[0] ** 2

    result = mskopt.dummy_minimize(objective, [(-1.0, 1.0)], n_calls=7, random_state=2)
    assert len(calls) == len(result.x_iters) == 7
    assert result.models == []


def test_gp_minimize_matches_upstream_optimization_quality():
    objective = lambda x: (x[0] - 0.23) ** 2 + 0.1 * (x[1] + 0.4) ** 2
    dimensions = [(-1.0, 1.0), (-1.0, 1.0)]
    ours = mskopt.gp_minimize(
        objective, dimensions, n_calls=18, n_initial_points=6, n_points=3000, random_state=5
    )
    theirs = skopt.gp_minimize(
        objective, dimensions, n_calls=18, n_initial_points=6, n_points=3000, random_state=5
    )
    assert ours.fun < 5e-3
    assert ours.fun <= theirs.fun + 5e-3
    assert len(ours.x_iters) == len(theirs.x_iters) == 18


def test_forest_minimize_matches_upstream_optimization_quality():
    objective = lambda x: (x[0] - 0.31) ** 2
    ours = mskopt.forest_minimize(
        objective, [(-1.0, 1.0)], n_calls=16, n_initial_points=5, n_points=2000, random_state=9
    )
    theirs = skopt.forest_minimize(
        objective, [(-1.0, 1.0)], n_calls=16, n_initial_points=5, n_points=2000, random_state=9
    )
    assert ours.fun < 2e-3
    assert ours.fun <= theirs.fun + 2e-3


def test_constraint_callback_and_named_args():
    dimensions = [Real(-1.0, 1.0, name="x"), Integer(0, 4, name="n")]

    @use_named_args(dimensions)
    def objective(x, n):
        return (x - 0.1) ** 2 + (n - 2) ** 2

    seen = []

    def stop_after_seven(result):
        seen.append(result.x_iters[-1])
        return len(seen) == 7

    result = mskopt.gp_minimize(
        objective,
        dimensions,
        n_calls=20,
        n_initial_points=4,
        n_points=1000,
        random_state=1,
        callback=stop_after_seven,
        space_constraint=lambda point: point[0] + point[1] / 10 <= 1.0,
    )
    assert len(seen) == 7
    assert len(result.x_iters) == 7
    assert all(point[0] + point[1] / 10 <= 1.0 for point in result.x_iters)
