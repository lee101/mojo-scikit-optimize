"""Common scikit-optimize utility functions."""

from __future__ import annotations

from collections import OrderedDict
from functools import wraps
import pickle

import numpy as np

from .space import Space


def use_named_args(dimensions):
    dimensions = Space(dimensions).dimensions
    names = [dimension.name for dimension in dimensions]
    if any(name is None for name in names):
        raise ValueError("All dimensions must have names")

    def decorator(func):
        @wraps(func)
        def wrapper(x):
            return func(**dict(zip(names, x)))

        return wrapper

    return decorator


def dimensions_aslist(search_space):
    return [
        value if getattr(value, "name", None) is not None else _set_name(value, key)
        for key, value in sorted(search_space.items())
    ]


def _set_name(dimension, name):
    dimension.name = name
    return dimension


def point_asdict(search_space, point):
    return OrderedDict(zip(sorted(search_space), point))


def expected_minimum_random_sampling(res, n_random_starts=100000, random_state=None):
    rng = np.random.RandomState(random_state)
    points = res.space.rvs(n_random_starts, random_state=rng)
    values = res.models[-1].predict(res.space.transform(points))
    index = int(np.argmin(values))
    return points[index], float(values[index])


def expected_minimum(res, n_random_starts=20, random_state=None):
    return expected_minimum_random_sampling(
        res, max(1000, n_random_starts * 100), random_state
    )


def dump(res, filename, store_objective=True, **kwargs):
    with open(filename, "wb") as stream:
        pickle.dump(res, stream, **kwargs)


def load(filename, **kwargs):
    with open(filename, "rb") as stream:
        return pickle.load(stream, **kwargs)
