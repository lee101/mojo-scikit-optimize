# mojo-scikit-optimize

Bayesian optimization with Mojo-accelerated Gaussian-process and forest
surrogates, exposed through a Python API matching the covered subset of
[scikit-optimize](https://scikit-optimize.readthedocs.io/).

This is a standalone implementation: importing `mojo_skopt` does not import or
delegate to scikit-optimize. The Pixi environment includes scikit-optimize
0.10.x so the test and benchmark suites can compare both implementations on the
same inputs.

```python
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
```

## Coverage

The table lists the upstream-shaped surface exercised by the parity suite.

| Upstream area | Tested coverage |
| --- | --- |
| Optimizers | `Optimizer` with `ask`, batched `ask`, `tell`, `copy`, and `get_result`; `gp_minimize`, `forest_minimize`, `dummy_minimize` |
| Search spaces | `Space`, `Real`, `Integer`, `Categorical`; uniform and log-uniform sampling; transforms and constraints |
| Gaussian processes | `GaussianProcessRegressor`; RBF and Matérn ν=0.5/1.5/2.5 covariance; normalization; scalar or per-sample noise; posterior mean, standard deviation, covariance, and single-point gradients |
| Forests | `ExtraTreesRegressor` and `RandomForestRegressor`; upstream-compatible conditional standard deviation and `min_variance` |
| Acquisition | `gaussian_ei`, `gaussian_pi`, and `gaussian_lcb`; batch values and single-point gradients |
| Utility | `use_named_args` |

The public call signatures of the three minimizers, `Optimizer`, and the
Gaussian expected-improvement function match scikit-optimize 0.10.2. Tests also
exercise non-SIMD-width Cholesky and GP prediction tails, covariance, GP, and
forest batch sizes, invalid and empty FFI inputs, and unsupported
kernel rejection.

Not covered are MES/PVRS and per-second acquisition functions, BayesSearchCV,
plotting, the full callback collection, low-discrepancy initial generators,
multi-output regressors, arbitrary custom GP kernels, and scikit-optimize's
L-BFGS acquisition optimizer. Candidate optimization here is random sampling;
`initial_point_generator` currently accepts the upstream argument but uses
random sampling. Supplied GP kernel hyperparameters are used as-is rather than
optimized. Utility functions other than `use_named_args` are small convenience
implementations and are not claimed as upstream-compatible. Forest training
uses scikit-learn's mature tree builder; fitted-tree batch traversal and
uncertainty reduction run in Mojo.

## Install

```bash
git clone https://github.com/lee101/mojo-scikit-optimize.git
cd mojo-scikit-optimize
pixi install
pixi run build
pixi run test
```

Run the usage example from the repository with
`pixi run python examples/basic.py`, or paste the snippet above into
`pixi run python`.

`pixi run build` compiles `src/capi.mojo` with `mojo build --emit shared-lib`
to `dist/libmojo-scikit-optimize.so`. The Python binding also rebuilds a missing
or stale library on first use. `pixi run bench` is the only supported benchmark
entry point because its Pixi task takes a machine-wide lock.

## Performance

Measured on this machine on 2026-08-27 with `pixi run bench`: Intel Xeon
E5-2697 v4, Linux x86_64. Times are the best of three warm runs.

| case | mojo-scikit-optimize | scikit-optimize | result |
| --- | ---: | ---: | ---: |
| Matérn covariance (3000 x 500 x 8) | 12.26 ms | 389.51 ms | 31.78x faster |
| Expected improvement (2M) | 73.99 ms | 1310.47 ms | 17.71x faster |
| GP.fit (650 x 6) | 34.41 ms | 88.24 ms | 2.56x faster |
| GP.predict mean+std (160 train, 20k query) | 29.88 ms | 1344.39 ms | 45.00x faster |
| ExtraTrees.predict mean+std (100 trees, 100k) | 582.10 ms | 7100.57 ms | 12.20x faster |

GP fitting uses hardware-width float64 SIMD for Cholesky dot products and the
forward solve, including scalar remainder loops. Every kernel is single-threaded:
Mojo 1.2 removed closure capture, so a state-carrying kernel can no longer be
fanned out from inside a `parallelize` body. Each kernel was re-checked against
its arithmetic intensity instead. GP covariance reaches roughly two flops per
byte, right at the break-even point, and a measurement on this machine (3.2M to
13M covariance pairs, 2 and 4 threads) returned 2.1x to 2.5x at the very top of
that range but a loss at the 160k-pair sizes a Gaussian-process surrogate
actually runs at, so the kernel stays serial. The GP triangular solve runs at
about 0.25 flops per byte and forest traversal at about 0.09, both clearly
memory-bound. GP prediction reuses one scratch row for the whole query batch
instead of allocating one per query. Its scikit-learn-compatible float32
feature buffer crosses the FFI directly instead of being copied back to
float64.

No GPU path or GPU performance claim is included. Covariance distance work,
triangular solves, and branch-heavy forest traversal do not exceed the roughly
two-flops-per-byte threshold once their input loads are counted, so PCIe copies
and launch overhead would not be justified.

## How it works

Python owns estimator state, allocation, search-space handling, objective
calls, and the optimizer loop. NumPy inputs become C-contiguous arrays with the
kernel's required dtype. ctypes passes each buffer as an integer address plus
explicit extents; the exported Mojo function reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the ABI boundary.

```text
python/mojo_skopt/       upstream-shaped Python API and estimator state
          |
          | ctypes: Int addresses, dimensions, scalar options
          v
src/capi.mojo            one shared-library compilation unit
          |
          +-- RBF/Matérn covariance and GP prediction
          +-- in-place Cholesky and triangular solves
          +-- EI/PI/LCB batch evaluation
          +-- packed forest traversal and uncertainty moments
```

Matrices are row-major. GP buffers cross as contiguous float64 arrays; forest
features cross as scikit-learn-compatible contiguous float32 arrays. The GP
predictor writes an `m × n_train` caller-owned scratch matrix so Mojo performs
no heap allocation. Forest nodes from all scikit-learn estimators are
concatenated into contiguous child, feature, threshold, leaf-value, and
impurity arrays; child indices are rebased once when the model is fitted.

## License

MIT
