"""Sequential Bayesian optimization with GP and forest surrogates."""

from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import OptimizeResult
from sklearn.base import clone
from sklearn.utils import check_random_state

from .acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
from .learning import ExtraTreesRegressor, GaussianProcessRegressor, RandomForestRegressor
from .learning.gaussian_process.kernels import ConstantKernel, Matern
from .space import Space


def _point_list(value, n_dims):
    if value is None:
        return []
    if len(value) == n_dims and (n_dims == 0 or not isinstance(value[0], (list, tuple, np.ndarray))):
        return [list(value)]
    return [list(point) for point in value]


class Optimizer:
    def __init__(
        self,
        dimensions,
        base_estimator="gp",
        n_random_starts=None,
        n_initial_points=10,
        initial_point_generator="random",
        n_jobs=1,
        acq_func="gp_hedge",
        acq_optimizer="auto",
        random_state=None,
        model_queue_size=None,
        space_constraint=None,
        acq_func_kwargs=None,
        acq_optimizer_kwargs=None,
        avoid_duplicates=True,
    ):
        self.space = Space(dimensions, constraint=space_constraint)
        self.rng = check_random_state(random_state)
        self.base_estimator = base_estimator
        self.n_initial_points_ = (
            int(n_random_starts) if n_random_starts is not None else int(n_initial_points)
        )
        self.initial_point_generator = initial_point_generator
        self.n_jobs = n_jobs
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.model_queue_size = model_queue_size
        self.acq_func_kwargs = acq_func_kwargs or {}
        self.acq_optimizer_kwargs = acq_optimizer_kwargs or {}
        self.avoid_duplicates = avoid_duplicates
        self.Xi = []
        self.yi = []
        self.models = []
        self._model = None

    def _new_model(self):
        if not isinstance(self.base_estimator, str):
            return clone(self.base_estimator)
        name = self.base_estimator.upper()
        if name in {"DUMMY", "NONE"}:
            return None
        if name in {"GP", "GAUSSIANPROCESSREGRESSOR"}:
            dimensions = self.space.transformed_n_dims
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
                np.ones(dimensions), length_scale_bounds="fixed", nu=2.5
            )
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-10,
                optimizer=None,
                normalize_y=True,
                noise="gaussian",
                random_state=self.rng,
            )
        if name in {"ET", "EXTRATREESREGRESSOR"}:
            return ExtraTreesRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                min_variance=1e-12,
                n_jobs=self.n_jobs,
                random_state=self.rng,
            )
        if name in {"RF", "RANDOMFORESTREGRESSOR"}:
            return RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                min_variance=1e-12,
                n_jobs=self.n_jobs,
                random_state=self.rng,
            )
        raise ValueError(f"unknown base_estimator {self.base_estimator!r}")

    def _fit_model(self):
        model = self._new_model()
        if model is None:
            self._model = None
            return
        model.fit(self.space.transform(self.Xi), np.asarray(self.yi))
        self._model = model
        self.models.append(model)
        if self.model_queue_size is not None:
            self.models[:] = self.models[-int(self.model_queue_size) :]

    def _random_point(self, excluded=()):
        excluded_keys = {repr(point) for point in excluded}
        for _ in range(1000):
            point = self.space.rvs(1, random_state=self.rng)[0]
            if not self.avoid_duplicates or repr(point) not in excluded_keys:
                return point
        return self.space.rvs(1, random_state=self.rng)[0]

    def _ask_one(self, excluded=()):
        seen = self.Xi + list(excluded)
        if len(self.Xi) < self.n_initial_points_ or self._model is None:
            return self._random_point(seen)
        n_points = int(self.acq_optimizer_kwargs.get("n_points", 10000))
        candidates = self.space.rvs(n_points, random_state=self.rng)
        transformed = self.space.transform(candidates)
        y_opt = float(np.min(self.yi))
        xi = float(self.acq_func_kwargs.get("xi", 0.01))
        kappa = float(self.acq_func_kwargs.get("kappa", 1.96))
        name = self.acq_func.upper()
        if name == "EI":
            score = gaussian_ei(transformed, self._model, y_opt=y_opt, xi=xi)
            order = np.argsort(-score)
        elif name == "PI":
            score = gaussian_pi(transformed, self._model, y_opt=y_opt, xi=xi)
            order = np.argsort(-score)
        elif name == "LCB":
            score = gaussian_lcb(transformed, self._model, kappa=kappa)
            order = np.argsort(score)
        elif name == "GP_HEDGE":
            ei = gaussian_ei(transformed, self._model, y_opt=y_opt, xi=xi)
            pi = gaussian_pi(transformed, self._model, y_opt=y_opt, xi=xi)
            lcb = gaussian_lcb(transformed, self._model, kappa=kappa)
            choices = [
                int(np.argmax(ei)),
                int(np.argmax(pi)),
                int(np.argmin(lcb)),
            ]
            predictions = self._model.predict(transformed[choices])
            order = np.array([choices[int(np.argmin(predictions))]])
        else:
            raise ValueError("acq_func must be EI, PI, LCB, or gp_hedge")
        seen_keys = {repr(point) for point in seen}
        for index in order:
            point = candidates[int(index)]
            if not self.avoid_duplicates or repr(point) not in seen_keys:
                return point
        return self._random_point(seen)

    def ask(self, n_points=None, strategy="cl_min"):
        if n_points is None:
            return self._ask_one()
        if strategy not in {"cl_min", "cl_mean", "cl_max"}:
            raise ValueError("strategy must be cl_min, cl_mean, or cl_max")
        points = []
        for _ in range(int(n_points)):
            points.append(self._ask_one(points))
        return points

    def tell(self, x, y, fit=True):
        points = _point_list(x, len(self.space))
        values = np.asarray(y)
        if values.ndim == 0:
            values = values.reshape(1)
        if len(points) != len(values):
            raise ValueError("x and y must contain the same number of observations")
        self.Xi.extend(copy.deepcopy(points))
        self.yi.extend(float(value) for value in values)
        if fit and len(self.Xi) >= self.n_initial_points_:
            self._fit_model()
        return self.get_result()

    def get_result(self):
        result = OptimizeResult()
        result.x_iters = copy.deepcopy(self.Xi)
        result.func_vals = np.asarray(self.yi)
        if self.yi:
            index = int(np.argmin(self.yi))
            result.x = copy.deepcopy(self.Xi[index])
            result.fun = float(self.yi[index])
        else:
            result.x = None
            result.fun = np.inf
        result.models = list(self.models)
        result.space = self.space
        result.random_state = self.rng
        result.specs = {"args": {}, "function": None}
        return result

    def copy(self, random_state=None):
        duplicate = copy.deepcopy(self)
        if random_state is not None:
            duplicate.rng = check_random_state(random_state)
        return duplicate


def _run_minimize(
    func,
    dimensions,
    *,
    base_estimator,
    n_calls,
    n_initial_points,
    n_random_starts,
    initial_point_generator,
    acq_func,
    acq_optimizer,
    random_state,
    callback,
    n_points,
    xi,
    kappa,
    n_jobs,
    model_queue_size,
    space_constraint,
    x0,
    y0,
):
    optimizer = Optimizer(
        dimensions,
        base_estimator=base_estimator,
        n_random_starts=n_random_starts,
        n_initial_points=n_initial_points,
        initial_point_generator=initial_point_generator,
        n_jobs=n_jobs,
        acq_func=acq_func,
        acq_optimizer=acq_optimizer,
        random_state=random_state,
        model_queue_size=model_queue_size,
        space_constraint=space_constraint,
        acq_func_kwargs={"xi": xi, "kappa": kappa},
        acq_optimizer_kwargs={"n_points": n_points},
    )
    callbacks = [] if callback is None else (
        list(callback) if isinstance(callback, (list, tuple)) else [callback]
    )
    initial = _point_list(x0, len(optimizer.space))
    if initial:
        if y0 is None:
            values = [func(point) for point in initial]
        else:
            values = np.asarray(y0).reshape(-1)
        optimizer.tell(initial, values)
    remaining = int(n_calls) - len(initial)
    if remaining < 0:
        raise ValueError("n_calls must be at least the number of x0 points")
    for _ in range(remaining):
        point = optimizer.ask()
        result = optimizer.tell(point, func(point))
        if any(bool(item(result)) for item in callbacks):
            break
    return optimizer.get_result()


def gp_minimize(
    func,
    dimensions,
    base_estimator=None,
    n_calls=100,
    n_random_starts=None,
    n_initial_points=10,
    initial_point_generator="random",
    acq_func="gp_hedge",
    acq_optimizer="auto",
    x0=None,
    y0=None,
    random_state=None,
    verbose=False,
    callback=None,
    n_points=10000,
    n_restarts_optimizer=5,
    xi=0.01,
    kappa=1.96,
    noise="gaussian",
    n_jobs=1,
    model_queue_size=None,
    space_constraint=None,
):
    return _run_minimize(
        func,
        dimensions,
        base_estimator="GP" if base_estimator is None else base_estimator,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        n_random_starts=n_random_starts,
        initial_point_generator=initial_point_generator,
        acq_func=acq_func,
        acq_optimizer=acq_optimizer,
        random_state=random_state,
        callback=callback,
        n_points=n_points,
        xi=xi,
        kappa=kappa,
        n_jobs=n_jobs,
        model_queue_size=model_queue_size,
        space_constraint=space_constraint,
        x0=x0,
        y0=y0,
    )


def forest_minimize(
    func,
    dimensions,
    base_estimator="ET",
    n_calls=100,
    n_random_starts=None,
    n_initial_points=10,
    acq_func="EI",
    initial_point_generator="random",
    x0=None,
    y0=None,
    random_state=None,
    verbose=False,
    callback=None,
    n_points=10000,
    xi=0.01,
    kappa=1.96,
    n_jobs=1,
    model_queue_size=None,
    space_constraint=None,
):
    return _run_minimize(
        func,
        dimensions,
        base_estimator=base_estimator,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        n_random_starts=n_random_starts,
        initial_point_generator=initial_point_generator,
        acq_func=acq_func,
        acq_optimizer="sampling",
        random_state=random_state,
        callback=callback,
        n_points=n_points,
        xi=xi,
        kappa=kappa,
        n_jobs=n_jobs,
        model_queue_size=model_queue_size,
        space_constraint=space_constraint,
        x0=x0,
        y0=y0,
    )


def dummy_minimize(
    func,
    dimensions,
    n_calls=100,
    initial_point_generator="random",
    x0=None,
    y0=None,
    random_state=None,
    verbose=False,
    callback=None,
    model_queue_size=None,
    init_point_gen_kwargs=None,
    space_constraint=None,
):
    return _run_minimize(
        func,
        dimensions,
        base_estimator="DUMMY",
        n_calls=n_calls,
        n_initial_points=n_calls,
        n_random_starts=None,
        initial_point_generator=initial_point_generator,
        acq_func="EI",
        acq_optimizer="sampling",
        random_state=random_state,
        callback=callback,
        n_points=1,
        xi=0.01,
        kappa=1.96,
        n_jobs=1,
        model_queue_size=model_queue_size,
        space_constraint=space_constraint,
        x0=x0,
        y0=y0,
    )
