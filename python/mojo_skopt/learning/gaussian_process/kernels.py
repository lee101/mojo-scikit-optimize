"""Gaussian-process kernels accepted by the Mojo regressor."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Exponentiation,
    Hyperparameter,
    Kernel,
    Matern,
    Product,
    RBF,
    RationalQuadratic,
    Sum,
    WhiteKernel,
)


class HammingKernel(Kernel):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        anisotropic = np.iterable(self.length_scale) and len(np.atleast_1d(self.length_scale)) > 1
        return Hyperparameter(
            "length_scale",
            "numeric",
            self.length_scale_bounds,
            len(np.atleast_1d(self.length_scale)) if anisotropic else 1,
        )

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        scale = np.asarray(self.length_scale)
        kernel = np.exp(-np.sum((X[:, None, :] != Y[None, :, :]) / scale, axis=2))
        if eval_gradient:
            if Y is not X:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            gradient = kernel[:, :, None] * np.sum(
                (X[:, None, :] != X[None, :, :]) / scale, axis=2
            )[:, :, None]
            return kernel, gradient
        return kernel

    def diag(self, X):
        return np.ones(np.asarray(X).shape[0])

    def is_stationary(self):
        return True


__all__ = [
    "ConstantKernel",
    "DotProduct",
    "ExpSineSquared",
    "Exponentiation",
    "HammingKernel",
    "Hyperparameter",
    "Kernel",
    "Matern",
    "Product",
    "RBF",
    "RationalQuadratic",
    "Sum",
    "WhiteKernel",
]
