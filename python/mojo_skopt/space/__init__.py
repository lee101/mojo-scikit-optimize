"""Search-space dimensions compatible with :mod:`skopt.space`."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.utils import check_random_state


class Dimension:
    name = None

    def rvs(self, n_samples=1, random_state=None):
        raise NotImplementedError

    def transform(self, values):
        raise NotImplementedError

    def inverse_transform(self, values):
        raise NotImplementedError

    def distance(self, a, b):
        return float(a != b)


class Real(Dimension):
    def __init__(
        self,
        low,
        high,
        prior="uniform",
        base=10,
        transform=None,
        name=None,
        dtype=float,
    ):
        if high <= low:
            raise ValueError("high must be greater than low")
        if prior not in {"uniform", "log-uniform"}:
            raise ValueError("prior must be 'uniform' or 'log-uniform'")
        self.low = float(low)
        self.high = float(high)
        self.prior = prior
        self.base = base
        self.transform_ = transform
        self.name = name
        self.dtype = dtype

    @property
    def bounds(self):
        return (self.low, self.high)

    @property
    def transformed_size(self):
        return 1

    def rvs(self, n_samples=1, random_state=None):
        rng = check_random_state(random_state)
        if self.prior == "log-uniform":
            low, high = np.log(self.low) / np.log(self.base), np.log(self.high) / np.log(self.base)
            return (self.base ** rng.uniform(low, high, n_samples)).astype(self.dtype).tolist()
        return rng.uniform(self.low, self.high, n_samples).astype(self.dtype).tolist()

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        if self.prior == "log-uniform":
            values = np.log(values) / np.log(self.base)
            low = np.log(self.low) / np.log(self.base)
            high = np.log(self.high) / np.log(self.base)
        else:
            low, high = self.low, self.high
        return ((values - low) / (high - low))[:, None]

    def inverse_transform(self, values):
        values = np.clip(np.asarray(values, dtype=float).reshape(-1), 0.0, 1.0)
        if self.prior == "log-uniform":
            low = np.log(self.low) / np.log(self.base)
            high = np.log(self.high) / np.log(self.base)
            values = self.base ** (low + values * (high - low))
        else:
            values = self.low + values * (self.high - self.low)
        return values.astype(self.dtype).tolist()

    def distance(self, a, b):
        return abs(float(a) - float(b))


class Integer(Dimension):
    def __init__(
        self,
        low,
        high,
        prior="uniform",
        base=10,
        transform=None,
        name=None,
        dtype=np.int64,
    ):
        if high < low:
            raise ValueError("high must be at least low")
        self.low = int(low)
        self.high = int(high)
        self.prior = prior
        self.base = base
        self.transform_ = transform
        self.name = name
        self.dtype = dtype

    @property
    def bounds(self):
        return (self.low, self.high)

    @property
    def transformed_size(self):
        return 1

    def rvs(self, n_samples=1, random_state=None):
        rng = check_random_state(random_state)
        if self.prior == "log-uniform":
            low = np.log(self.low) / np.log(self.base)
            high = np.log(self.high) / np.log(self.base)
            values = np.rint(self.base ** rng.uniform(low, high, n_samples))
            return np.clip(values, self.low, self.high).astype(self.dtype).tolist()
        return rng.randint(self.low, self.high + 1, n_samples).astype(self.dtype).tolist()

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        if self.high == self.low:
            return np.zeros((len(values), 1))
        return ((values - self.low) / (self.high - self.low))[:, None]

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        result = np.rint(self.low + np.clip(values, 0.0, 1.0) * (self.high - self.low))
        return result.astype(self.dtype).tolist()

    def distance(self, a, b):
        return abs(int(a) - int(b))


class Categorical(Dimension):
    def __init__(self, categories, prior=None, transform=None, name=None):
        self.categories = tuple(categories)
        if not self.categories:
            raise ValueError("categories cannot be empty")
        self.prior = prior
        self.transform_ = transform
        self.name = name

    @property
    def bounds(self):
        return self.categories

    @property
    def transformed_size(self):
        return len(self.categories)

    def rvs(self, n_samples=1, random_state=None):
        rng = check_random_state(random_state)
        probabilities = None if self.prior is None else np.asarray(self.prior, dtype=float)
        return [self.categories[i] for i in rng.choice(len(self.categories), n_samples, p=probabilities)]

    def transform(self, values):
        result = np.zeros((len(values), len(self.categories)), dtype=float)
        index = {value: i for i, value in enumerate(self.categories)}
        for row, value in enumerate(values):
            if value not in index:
                raise ValueError(f"{value!r} is not in the categories")
            result[row, index[value]] = 1.0
        return result

    def inverse_transform(self, values):
        values = np.atleast_2d(values)
        return [self.categories[index] for index in np.argmax(values, axis=1)]


def check_dimension(dimension):
    if isinstance(dimension, Dimension):
        return dimension
    if not isinstance(dimension, (tuple, list)):
        raise ValueError(f"Invalid dimension {dimension!r}")
    if len(dimension) == 2 and all(isinstance(value, numbers.Integral) for value in dimension):
        return Integer(*dimension)
    if len(dimension) in {2, 3} and all(
        isinstance(value, numbers.Real) for value in dimension[:2]
    ):
        return Real(*dimension)
    return Categorical(dimension)


class Space:
    def __init__(self, dimensions, constraint=None):
        self.dimensions = [check_dimension(dimension) for dimension in dimensions]
        self.constraint = constraint

    def __len__(self):
        return len(self.dimensions)

    def __getitem__(self, index):
        return self.dimensions[index]

    @property
    def bounds(self):
        return [dimension.bounds for dimension in self.dimensions]

    @property
    def transformed_n_dims(self):
        return sum(dimension.transformed_size for dimension in self.dimensions)

    @property
    def dimension_names(self):
        return [dimension.name for dimension in self.dimensions]

    def rvs(self, n_samples=1, random_state=None):
        rng = check_random_state(random_state)
        points = []
        attempts = 0
        while len(points) < n_samples:
            point = [dimension.rvs(1, rng)[0] for dimension in self.dimensions]
            if self.constraint is None or self.constraint(point):
                points.append(point)
            attempts += 1
            if attempts > max(10000, 1000 * n_samples):
                raise RuntimeError("could not sample enough points satisfying the constraint")
        return points

    def transform(self, X):
        X = list(X)
        if not X:
            return np.empty((0, self.transformed_n_dims))
        columns = list(zip(*X))
        return np.ascontiguousarray(
            np.concatenate(
                [
                    dimension.transform(list(values))
                    for dimension, values in zip(self.dimensions, columns)
                ],
                axis=1,
            ),
            dtype=np.float64,
        )

    def inverse_transform(self, Xt):
        Xt = np.atleast_2d(Xt)
        columns = []
        start = 0
        for dimension in self.dimensions:
            stop = start + dimension.transformed_size
            columns.append(dimension.inverse_transform(Xt[:, start:stop]))
            start = stop
        return [list(point) for point in zip(*columns)]

    def distance(self, point_a, point_b):
        return sum(
            dimension.distance(a, b)
            for dimension, a, b in zip(self.dimensions, point_a, point_b)
        )


__all__ = ["Categorical", "Dimension", "Integer", "Real", "Space", "check_dimension"]
