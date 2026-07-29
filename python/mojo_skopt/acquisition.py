"""Gaussian acquisition functions compatible with :mod:`skopt.acquisition`."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm

from ._lib import addr, f64, lib


def _posterior(X, model, return_grad):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if return_grad:
            return model.predict(
                X, return_std=True, return_mean_grad=True, return_std_grad=True
            )
        return model.predict(X, return_std=True)


def _values(mu, std, kind, y_opt=0.0, xi=0.01, kappa=1.96):
    mu = f64(mu)
    std = f64(std)
    if mu.ndim != 1 or std.ndim != 1:
        raise ValueError("mu and std must both be one-dimensional")
    if mu.shape != std.shape:
        raise ValueError("mu and std must have the same length")
    if len(mu) == 0:
        return np.empty(0, dtype=np.float64)
    values = np.empty_like(mu)
    lib().msko_acquisition(
        addr(mu), addr(std), addr(values), len(mu), kind, y_opt, xi, kappa
    )
    return values


def gaussian_lcb(X, model, kappa=1.96, return_grad=False):
    posterior = _posterior(X, model, return_grad)
    if return_grad:
        mu, std, mu_grad, std_grad = posterior
        if kappa == "inf":
            return -std, -std_grad
        return mu - kappa * std, mu_grad - kappa * std_grad
    mu, std = posterior
    if kappa == "inf":
        return -std
    return _values(mu, std, 0, kappa=float(kappa))


def gaussian_pi(X, model, y_opt=0.0, xi=0.01, return_grad=False):
    posterior = _posterior(X, model, return_grad)
    if not return_grad:
        return _values(*posterior, 1, y_opt=float(y_opt), xi=float(xi))
    mu, std, mu_grad, std_grad = posterior
    if mu.ndim != 1 or std.ndim != 1:
        raise ValueError("mu and std must both be one-dimensional")
    values = _values(mu, std, 1, y_opt=float(y_opt), xi=float(xi))
    mask = std > 0
    if not np.all(mask):
        return values, np.zeros_like(std_grad)
    improve = y_opt - xi - mu
    scaled = improve / std
    scaled_grad = (-mu_grad * std - std_grad * improve) / std**2
    return values, scaled_grad * norm.pdf(scaled)


def gaussian_ei(X, model, y_opt=0.0, xi=0.01, return_grad=False):
    posterior = _posterior(X, model, return_grad)
    if not return_grad:
        return _values(*posterior, 2, y_opt=float(y_opt), xi=float(xi))
    mu, std, mu_grad, std_grad = posterior
    if mu.ndim != 1 or std.ndim != 1:
        raise ValueError("mu and std must both be one-dimensional")
    values = _values(mu, std, 2, y_opt=float(y_opt), xi=float(xi))
    mask = std > 0
    if not np.all(mask):
        return values, np.zeros_like(std_grad)
    improve = y_opt - xi - mu
    scaled = improve / std
    return values, -mu_grad * norm.cdf(scaled) + std_grad * norm.pdf(scaled)


def _gaussian_acquisition(
    X, model, y_opt=None, acq_func="LCB", return_grad=False, acq_func_kwargs=None
):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X is {X.ndim}-dimensional, however, it must be 2-dimensional.")
    kwargs = acq_func_kwargs or {}
    if acq_func == "LCB":
        return gaussian_lcb(X, model, kwargs.get("kappa", 1.96), return_grad)
    if acq_func in {"EI", "EIps"}:
        result = gaussian_ei(X, model, y_opt, kwargs.get("xi", 0.01), return_grad)
    elif acq_func in {"PI", "PIps"}:
        result = gaussian_pi(X, model, y_opt, kwargs.get("xi", 0.01), return_grad)
    else:
        raise ValueError("Acquisition function not implemented.")
    if return_grad:
        return -result[0], -result[1]
    return -result


def gaussian_acquisition_1D(
    X, model, y_opt=None, acq_func="LCB", acq_func_kwargs=None, return_grad=True
):
    return _gaussian_acquisition(
        np.expand_dims(X, axis=0),
        model,
        y_opt,
        acq_func=acq_func,
        acq_func_kwargs=acq_func_kwargs,
        return_grad=return_grad,
    )
