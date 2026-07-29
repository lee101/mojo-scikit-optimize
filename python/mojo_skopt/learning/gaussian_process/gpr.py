"""Gaussian-process regression with Mojo covariance and prediction kernels."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.utils.validation import check_is_fitted

from ..._lib import addr, f64, lib


def _kernel_parameters(kernel, n_features):
    amplitude, noise = 1.0, 0.0
    parts = [kernel]
    if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
        if kernel.__class__.__name__ != "Product":
            raise ValueError("only products of ConstantKernel and RBF/Matérn are supported")
        parts = [kernel.k1, kernel.k2]

    covariance = None
    for part in parts:
        if isinstance(part, ConstantKernel):
            amplitude *= float(part.constant_value)
        elif isinstance(part, WhiteKernel):
            noise += float(part.noise_level)
        elif isinstance(part, (Matern, RBF)):
            if covariance is not None:
                raise ValueError("exactly one RBF or Matérn covariance kernel is supported")
            covariance = part
        else:
            raise ValueError(f"unsupported kernel component {type(part).__name__}")
    if covariance is None:
        raise ValueError("an RBF or Matérn covariance kernel is required")
    if isinstance(covariance, Matern):
        kinds = {0.5: 1, 1.5: 2, 2.5: 3}
        if float(covariance.nu) not in kinds:
            raise ValueError("Matérn nu must be 0.5, 1.5, or 2.5")
        kind = kinds[float(covariance.nu)]
    else:
        kind = 0
    value = np.asarray(covariance.length_scale, dtype=np.float64)
    length_scale = np.full(n_features, value.item()) if value.ndim == 0 else value
    if length_scale.shape != (n_features,):
        raise ValueError("length_scale must be scalar or have one value per feature")
    if not np.all(np.isfinite(length_scale)) or np.any(length_scale <= 0):
        raise ValueError("length_scale values must be finite and positive")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("kernel amplitude must be finite and positive")
    return f64(length_scale), amplitude, kind, noise


class GaussianProcessRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        kernel=None,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
        noise=None,
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.noise = noise

    def fit(self, X, y):
        X = f64(X, copy=self.copy_X_train)
        y = f64(y, copy=True).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-dimensional and have the same length as y")
        if len(X) == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values")
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X
        self._y_train_mean = float(np.mean(y)) if self.normalize_y else 0.0
        scale = float(np.std(y)) if self.normalize_y else 1.0
        self._y_train_std = scale if scale > 0 else 1.0
        self.y_train_ = f64((y - self._y_train_mean) / self._y_train_std)

        if self.kernel is None:
            self.kernel_ = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
        else:
            self.kernel_ = self.kernel
        self.length_scale_, self.amplitude_, self.kernel_kind_, kernel_noise = (
            _kernel_parameters(self.kernel_, self.n_features_in_)
        )
        if self.optimizer is not None and self.kernel is not None:
            warnings.warn(
                "Mojo GaussianProcessRegressor uses supplied kernel parameters without "
                "hyperparameter optimization; pass optimizer=None to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
        n = len(X)
        K = np.empty((n, n), dtype=np.float64)
        lib().msko_covariance(
            addr(X),
            addr(X),
            addr(K),
            n,
            n,
            X.shape[1],
            addr(self.length_scale_),
            self.amplitude_,
            self.kernel_kind_,
        )
        diagonal = np.asarray(self.alpha, dtype=float)
        if diagonal.ndim == 0:
            K.flat[:: n + 1] += float(diagonal) + kernel_noise
        elif diagonal.shape == (n,):
            K.flat[:: n + 1] += diagonal + kernel_noise
        else:
            raise ValueError("alpha must be a scalar or an array with one value per sample")
        if self.noise == "gaussian":
            K.flat[:: n + 1] += 1e-8
        self.L_ = K
        jitter = 0.0
        for attempt in range(7):
            candidate = self.L_.copy()
            if jitter:
                candidate.flat[:: n + 1] += jitter
            if lib().msko_cholesky(addr(candidate), n):
                self.L_ = candidate
                break
            jitter = 10.0 ** (-12 + attempt)
        else:
            raise np.linalg.LinAlgError("kernel matrix is not positive definite")
        self.alpha_ = self.y_train_.copy()
        lib().msko_cholesky_solve(addr(self.L_), addr(self.alpha_), n)
        self.log_marginal_likelihood_value_ = float(
            -0.5 * self.y_train_.dot(self.alpha_)
            - np.log(np.diag(self.L_)).sum()
            - 0.5 * n * np.log(2.0 * np.pi)
        )
        return self

    def _cross_covariance(self, X):
        result = np.empty((len(X), len(self.X_train_)), dtype=np.float64)
        lib().msko_covariance(
            addr(X),
            addr(self.X_train_),
            addr(result),
            len(X),
            len(self.X_train_),
            X.shape[1],
            addr(self.length_scale_),
            self.amplitude_,
            self.kernel_kind_,
        )
        return result

    def _gradient_x(self, x):
        delta = x[None, :] - self.X_train_
        scaled = delta / self.length_scale_
        r2 = np.sum(scaled**2, axis=1)
        r = np.sqrt(r2)
        if self.kernel_kind_ == 0:
            k = self.amplitude_ * np.exp(-0.5 * r2)
            return -k[:, None] * delta / self.length_scale_**2
        if self.kernel_kind_ == 1:
            factor = np.zeros_like(r)
            np.divide(
                -self.amplitude_ * np.exp(-r),
                r,
                out=factor,
                where=r > 0,
            )
        elif self.kernel_kind_ == 2:
            factor = -3.0 * self.amplitude_ * np.exp(-np.sqrt(3.0) * r)
        else:
            factor = (
                -(5.0 / 3.0)
                * self.amplitude_
                * (1.0 + np.sqrt(5.0) * r)
                * np.exp(-np.sqrt(5.0) * r)
            )
        return factor[:, None] * delta / self.length_scale_**2

    def predict(
        self,
        X,
        return_std=False,
        return_cov=False,
        return_mean_grad=False,
        return_std_grad=False,
    ):
        check_is_fitted(self, "X_train_")
        if return_std and return_cov:
            raise RuntimeError("Not returning standard deviation when returning covariance.")
        if return_std_grad and not return_std:
            raise ValueError("Not returning std_gradient without return_std.")
        X = f64(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong shape")
        if len(X) != 1 and (return_mean_grad or return_std_grad):
            raise ValueError("Gradients are implemented only for one sample.")
        if len(X) == 0:
            empty = np.empty(0, dtype=np.float64)
            if return_cov:
                return empty, np.empty((0, 0), dtype=np.float64)
            if return_std:
                return empty, empty.copy()
            return empty
        mean = np.empty(len(X), dtype=np.float64)
        std = np.empty(len(X), dtype=np.float64)
        work = np.empty((len(X), len(self.X_train_)), dtype=np.float64)
        lib().msko_gp_predict(
            addr(self.X_train_),
            addr(X),
            addr(self.alpha_),
            addr(self.L_),
            addr(self.length_scale_),
            addr(mean),
            addr(std),
            addr(work),
            len(self.X_train_),
            len(X),
            X.shape[1],
            self.amplitude_,
            self.kernel_kind_,
        )
        mean = mean * self._y_train_std + self._y_train_mean
        std = std * self._y_train_std
        if return_cov:
            cross = self._cross_covariance(X)
            solved = np.linalg.solve(self.L_, cross.T)
            covariance = np.empty((len(X), len(X)), dtype=np.float64)
            lib().msko_covariance(
                addr(X),
                addr(X),
                addr(covariance),
                len(X),
                len(X),
                X.shape[1],
                addr(self.length_scale_),
                self.amplitude_,
                self.kernel_kind_,
            )
            covariance = (covariance - solved.T @ solved) * self._y_train_std**2
            return mean, covariance
        if return_mean_grad or return_std_grad:
            kernel_grad = self._gradient_x(X[0])
            mean_grad = kernel_grad.T @ self.alpha_ * self._y_train_std
            if return_std_grad:
                cross = self._cross_covariance(X)[0]
                inverse_cross = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, cross))
                std_grad = np.zeros(self.n_features_in_)
                base_std = std[0] / self._y_train_std
                if base_std > 0:
                    std_grad = (
                        -(kernel_grad.T @ inverse_cross)
                        / base_std
                        * self._y_train_std
                    )
                return mean, std, mean_grad, std_grad
            if return_std:
                return mean, std, mean_grad
            return mean, mean_grad
        if return_std:
            return mean, std
        return mean

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        check_is_fitted(self, "X_train_")
        if eval_gradient:
            return self.log_marginal_likelihood_value_, np.zeros_like(
                np.asarray(self.kernel_.theta)
            )
        return self.log_marginal_likelihood_value_
