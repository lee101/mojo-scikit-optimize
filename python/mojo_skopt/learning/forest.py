"""scikit-optimize forest regressors with Mojo batch prediction."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor as _ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .._lib import addr, f64, i64, lib


class _MojoForestMixin:
    def fit(self, X, y, sample_weight=None):
        fitted = super().fit(X, y, sample_weight=sample_weight)
        self._pack_trees()
        return fitted

    def _pack_trees(self):
        offsets = []
        left_parts = []
        right_parts = []
        feature_parts = []
        threshold_parts = []
        value_parts = []
        impurity_parts = []
        offset = 0
        for estimator in self.estimators_:
            tree = estimator.tree_
            offsets.append(offset)
            left = tree.children_left.astype(np.int64)
            right = tree.children_right.astype(np.int64)
            left[left >= 0] += offset
            right[right >= 0] += offset
            left_parts.append(left)
            right_parts.append(right)
            feature_parts.append(tree.feature.astype(np.int64))
            threshold_parts.append(tree.threshold.astype(np.float64))
            value_parts.append(tree.value[:, 0, 0].astype(np.float64))
            impurity_parts.append(tree.impurity.astype(np.float64))
            offset += tree.node_count
        self._mojo_offsets = i64(offsets)
        self._mojo_left = i64(np.concatenate(left_parts))
        self._mojo_right = i64(np.concatenate(right_parts))
        self._mojo_feature = i64(np.concatenate(feature_parts))
        self._mojo_threshold = f64(np.concatenate(threshold_parts))
        self._mojo_value = f64(np.concatenate(value_parts))
        self._mojo_impurity = f64(np.concatenate(impurity_parts))

    def predict(self, X, return_std=False):
        check_is_fitted(self, "estimators_")
        if self.n_outputs_ != 1:
            if return_std:
                raise ValueError("return_std is supported only for one output")
            return super().predict(X)
        if return_std and self.criterion != "squared_error":
            raise ValueError(
                "Expected impurity to be 'squared_error', got %s instead" % self.criterion
            )
        X = self._validate_X_predict(X)
        if sparse.issparse(X):
            X = X.toarray()
        X = np.ascontiguousarray(X, dtype=np.float32)
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if len(X) == 0:
            empty = np.empty(0, dtype=np.float64)
            return (empty, empty.copy()) if return_std else empty
        mean = np.empty(len(X), dtype=np.float64)
        std = np.empty(len(X), dtype=np.float64)
        lib().msko_forest_predict(
            addr(X),
            addr(self._mojo_offsets),
            addr(self._mojo_left),
            addr(self._mojo_right),
            addr(self._mojo_feature),
            addr(self._mojo_threshold),
            addr(self._mojo_value),
            addr(self._mojo_impurity),
            addr(mean),
            addr(std),
            len(X),
            X.shape[1],
            len(self.estimators_),
            float(self.min_variance),
        )
        return (mean, std) if return_std else mean


class RandomForestRegressor(_MojoForestMixin, _RandomForestRegressor):
    def __init__(
        self,
        n_estimators=10,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        min_variance=0.0,
    ):
        self.min_variance = min_variance
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )


class ExtraTreesRegressor(_MojoForestMixin, _ExtraTreesRegressor):
    def __init__(
        self,
        n_estimators=10,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        min_variance=0.0,
    ):
        self.min_variance = min_variance
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )
