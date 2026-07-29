"""ctypes binding for the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "dist" / "libmojo-scikit-optimize.so"
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msko_covariance": ([I, I, I, I, I, I, I, F, I], None),
    "msko_cholesky": ([I, I], I),
    "msko_cholesky_solve": ([I, I, I], None),
    "msko_gp_predict": ([I] * 11 + [F, I], None),
    "msko_acquisition": ([I, I, I, I, I, F, F, F], None),
    "msko_forest_predict": ([I] * 13 + [F], None),
}

_library: ctypes.CDLL | None = None


def build() -> Path:
    sources = list((ROOT / "src").glob("*.mojo"))
    stale = not LIB.exists() or any(path.stat().st_mtime > LIB.stat().st_mtime for path in sources)
    if stale:
        subprocess.run(["bash", str(ROOT / "build" / "build.sh")], cwd=ROOT, check=True)
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(os.fspath(build()))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    original = np.asarray(value)
    if np.iscomplexobj(original):
        raise TypeError("complex values cannot be narrowed to float64")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.float32), np.dtype(np.int64)):
        raise TypeError(f"unsupported FFI buffer dtype {array.dtype}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("empty or null buffers cannot cross the FFI boundary")
    return int(array.ctypes.data)
