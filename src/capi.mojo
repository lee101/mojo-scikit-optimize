"""Numerical kernels exported to Python through a small C ABI."""

from std.algorithm import parallelize
from std.math import erf, exp, sqrt
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime FOREST_PARALLEL_MIN_SAMPLES = 1024
comptime FOREST_PARALLEL_CHUNK = 64
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def scaled_sqdist(a: Ptr, b: Ptr, length_scale: Ptr, d: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= d:
        var diff = (a.load[width=W](i) - b.load[width=W](i)) / length_scale.load[width=W](i)
        acc += diff * diff
        i += W
    var total = acc.reduce_add()
    while i < d:
        var diff = (a[i] - b[i]) / length_scale[i]
        total += diff * diff
        i += 1
    return total


def covariance_value(
    a: Ptr, b: Ptr, length_scale: Ptr, d: Int, amplitude: Float64, kind: Int
) -> Float64:
    var r2 = scaled_sqdist(a, b, length_scale, d)
    if kind == 0:
        return amplitude * exp(-0.5 * r2)
    var r = sqrt(r2)
    if kind == 1:
        return amplitude * exp(-r)
    if kind == 2:
        var z = 1.7320508075688772 * r
        return amplitude * (1.0 + z) * exp(-z)
    var z = 2.2360679774997898 * r
    return amplitude * (1.0 + z + 1.6666666666666667 * r2) * exp(-z)


def covariance_matrix(
    x: Ptr,
    y: Ptr,
    dst: Ptr,
    n: Int,
    m: Int,
    d: Int,
    length_scale: Ptr,
    amplitude: Float64,
    kind: Int,
):
    for i in range(n):
        for j in range(m):
            dst[i * m + j] = covariance_value(
                x + i * d, y + j * d, length_scale, d, amplitude, kind
            )


def cholesky(a: Ptr, n: Int) -> Bool:
    for i in range(n):
        for j in range(i + 1):
            var value = a[i * n + j]
            var acc = SIMD[DType.float64, W](0.0)
            var k = 0
            while k + W <= j:
                acc += (
                    (a + i * n).load[width=W](k)
                    * (a + j * n).load[width=W](k)
                )
                k += W
            value -= acc.reduce_add()
            while k < j:
                value -= a[i * n + k] * a[j * n + k]
                k += 1
            if i == j:
                if value <= 0.0:
                    return False
                a[i * n + i] = sqrt(value)
            else:
                a[i * n + j] = value / a[j * n + j]
    for i in range(n):
        for j in range(i + 1, n):
            a[i * n + j] = 0.0
    return True


def cholesky_solve(l: Ptr, rhs: Ptr, n: Int):
    for i in range(n):
        var value = rhs[i]
        var acc = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= i:
            acc += (l + i * n).load[width=W](j) * rhs.load[width=W](j)
            j += W
        value -= acc.reduce_add()
        while j < i:
            value -= l[i * n + j] * rhs[j]
            j += 1
        rhs[i] = value / l[i * n + i]
    for ri in range(n):
        var i = n - 1 - ri
        var value = rhs[i]
        for j in range(i + 1, n):
            value -= l[j * n + i] * rhs[j]
        rhs[i] = value / l[i * n + i]


def gp_predict(
    train: Ptr,
    query: Ptr,
    alpha: Ptr,
    l: Ptr,
    length_scale: Ptr,
    mean_dst: Ptr,
    std_dst: Ptr,
    work: Ptr,
    n: Int,
    m: Int,
    d: Int,
    amplitude: Float64,
    kind: Int,
):
    for q in range(m):
        var row = work + q * n
        var mean = 0.0
        for i in range(n):
            var kval = covariance_value(
                query + q * d, train + i * d, length_scale, d, amplitude, kind
            )
            row[i] = kval
            mean += kval * alpha[i]
        mean_dst[q] = mean
        var variance = amplitude
        for i in range(n):
            var value = row[i]
            for j in range(i):
                value -= l[i * n + j] * row[j]
            value /= l[i * n + i]
            row[i] = value
            variance -= value * value
        std_dst[q] = sqrt(variance) if variance > 0.0 else 0.0


def normal_cdf(x: Float64) -> Float64:
    return 0.5 * (1.0 + erf(x * 0.7071067811865475))


def normal_pdf(x: Float64) -> Float64:
    return 0.3989422804014327 * exp(-0.5 * x * x)


def acquisition(
    mu: Ptr,
    sigma: Ptr,
    dst: Ptr,
    n: Int,
    kind: Int,
    y_opt: Float64,
    xi: Float64,
    kappa: Float64,
):
    for i in range(n):
        if kind == 0:
            dst[i] = mu[i] - kappa * sigma[i]
        elif sigma[i] > 0.0:
            var improve = y_opt - xi - mu[i]
            var z = improve / sigma[i]
            if kind == 1:
                dst[i] = normal_cdf(z)
            else:
                dst[i] = improve * normal_cdf(z) + sigma[i] * normal_pdf(z)
        else:
            dst[i] = 0.0


def forest_predict(
    x: FPtr,
    offsets: IPtr,
    left: IPtr,
    right: IPtr,
    feature: IPtr,
    threshold: Ptr,
    value: Ptr,
    impurity: Ptr,
    mean_dst: Ptr,
    std_dst: Ptr,
    n_samples: Int,
    n_features: Int,
    n_trees: Int,
    min_variance: Float64,
):
    @parameter
    def predict_chunk(chunk: Int):
        var begin = chunk * FOREST_PARALLEL_CHUNK
        var end = min(begin + FOREST_PARALLEL_CHUNK, n_samples)
        for sample in range(begin, end):
            var mean_acc = 0.0
            var second_acc = 0.0
            for tree in range(n_trees):
                var node = Int(offsets[tree])
                while feature[node] >= 0:
                    var f = Int(feature[node])
                    if Float64(x[sample * n_features + f]) <= threshold[node]:
                        node = Int(left[node])
                    else:
                        node = Int(right[node])
                var prediction = value[node]
                var variance = impurity[node]
                if variance < min_variance:
                    variance = min_variance
                mean_acc += prediction
                second_acc += variance + prediction * prediction
            var mean = mean_acc / Float64(n_trees)
            var variance = second_acc / Float64(n_trees) - mean * mean
            mean_dst[sample] = mean
            std_dst[sample] = sqrt(variance) if variance > 0.0 else 0.0

    var chunks = (n_samples + FOREST_PARALLEL_CHUNK - 1) // FOREST_PARALLEL_CHUNK
    if n_samples >= FOREST_PARALLEL_MIN_SAMPLES:
        parallelize[predict_chunk](chunks, 16)
    else:
        for chunk in range(chunks):
            predict_chunk(chunk)


@export("msko_covariance")
def msko_covariance(
    x: Int,
    y: Int,
    dst: Int,
    n: Int,
    m: Int,
    d: Int,
    length_scale: Int,
    amplitude: Float64,
    kind: Int,
) abi("C"):
    covariance_matrix(p(x), p(y), p(dst), n, m, d, p(length_scale), amplitude, kind)


@export("msko_cholesky")
def msko_cholesky(a: Int, n: Int) abi("C") -> Int:
    return 1 if cholesky(p(a), n) else 0


@export("msko_cholesky_solve")
def msko_cholesky_solve(l: Int, rhs: Int, n: Int) abi("C"):
    cholesky_solve(p(l), p(rhs), n)


@export("msko_gp_predict")
def msko_gp_predict(
    train: Int,
    query: Int,
    alpha: Int,
    l: Int,
    length_scale: Int,
    mean_dst: Int,
    std_dst: Int,
    work: Int,
    n: Int,
    m: Int,
    d: Int,
    amplitude: Float64,
    kind: Int,
) abi("C"):
    gp_predict(
        p(train),
        p(query),
        p(alpha),
        p(l),
        p(length_scale),
        p(mean_dst),
        p(std_dst),
        p(work),
        n,
        m,
        d,
        amplitude,
        kind,
    )


@export("msko_acquisition")
def msko_acquisition(
    mu: Int,
    sigma: Int,
    dst: Int,
    n: Int,
    kind: Int,
    y_opt: Float64,
    xi: Float64,
    kappa: Float64,
) abi("C"):
    acquisition(p(mu), p(sigma), p(dst), n, kind, y_opt, xi, kappa)


@export("msko_forest_predict")
def msko_forest_predict(
    x: Int,
    offsets: Int,
    left: Int,
    right: Int,
    feature: Int,
    threshold: Int,
    value: Int,
    impurity: Int,
    mean_dst: Int,
    std_dst: Int,
    n_samples: Int,
    n_features: Int,
    n_trees: Int,
    min_variance: Float64,
) abi("C"):
    forest_predict(
        fp(x),
        ip(offsets),
        ip(left),
        ip(right),
        ip(feature),
        p(threshold),
        p(value),
        p(impurity),
        p(mean_dst),
        p(std_dst),
        n_samples,
        n_features,
        n_trees,
        min_variance,
    )
