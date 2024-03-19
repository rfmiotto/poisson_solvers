import numba as nb
import numpy as np


@nb.njit()
def laplacian(arr, h):
    dsdx, dsdy = gradient(arr, h)
    part1, _ = gradient(dsdx, h)
    _, part2 = gradient(dsdy, h)
    return part1 + part2


@nb.njit()
def TDMAsolver(a, b, c, d):
    """Thomas algorithm to solve tridiagonal linear systems with
    non-periodic BC.

    | b0  c0                 | | . |     | . |
    | a1  b1  c1             | | . |     | . |
    |     a2  b2  c2         | | x |  =  | d |
    |         ..........     | | . |     | . |
    |             an  bn  cn | | . |     | . |
    """
    n = len(b)

    cp = np.zeros(n)
    cp[0] = c[0] / b[0]
    for i in range(1, n - 1):
        cp[i] = c[i] / (b[i] - a[i] * cp[i - 1])

    dp = np.zeros(n)
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        dp[i] = (d[i] - a[i] * dp[i - 1]) / (b[i] - a[i] * cp[i - 1])

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


@nb.njit()
def compact_scheme_6th(vec, h):
    """
    6th Order compact finite difference scheme (non-periodic BC).

    Lele, S. K. - Compact finite difference schemes with spectral-like
    resolution. Journal of Computational Physics 103 (1992) 16-42
    """
    n = len(vec)
    rhs = np.zeros(n)

    a = 14.0 / 18.0
    b = 1.0 / 36.0

    rhs[2:-2] = (vec[3:-1] - vec[1:-3]) * (a / h) + (vec[4:] - vec[0:-4]) * (b / h)

    # boundaries:
    rhs[0] = (
        (-197.0 / 60.0) * vec[0]
        + (-5.0 / 12.0) * vec[1]
        + 5.0 * vec[2]
        + (-5.0 / 3.0) * vec[3]
        + (5.0 / 12.0) * vec[4]
        + (-1.0 / 20.0) * vec[5]
    ) / h

    rhs[1] = (
        (-20.0 / 33.0) * vec[0]
        + (-35.0 / 132.0) * vec[1]
        + (34.0 / 33.0) * vec[2]
        + (-7.0 / 33.0) * vec[3]
        + (2.0 / 33.0) * vec[4]
        + (-1.0 / 132.0) * vec[5]
    ) / h

    rhs[-1] = (
        (197.0 / 60.0) * vec[-1]
        + (5.0 / 12.0) * vec[-2]
        + (-5.0) * vec[-3]
        + (5.0 / 3.0) * vec[-4]
        + (-5.0 / 12.0) * vec[-5]
        + (1.0 / 20.0) * vec[-6]
    ) / h

    rhs[-2] = (
        (20.0 / 33.0) * vec[-1]
        + (35.0 / 132.0) * vec[-2]
        + (-34.0 / 33.0) * vec[-3]
        + (7.0 / 33.0) * vec[-4]
        + (-2.0 / 33.0) * vec[-5]
        + (1.0 / 132.0) * vec[-6]
    ) / h

    alpha1 = 5.0  # j = 1 and n
    alpha2 = 2.0 / 11  # j = 2 and n-1
    alpha = 1.0 / 3.0

    Db = np.ones(n)
    Da = alpha * np.ones(n)
    Dc = alpha * np.ones(n)

    # boundaries:
    Da[1] = alpha2
    Da[-1] = alpha1
    Da[-2] = alpha2
    Dc[0] = alpha1
    Dc[1] = alpha2
    Dc[-2] = alpha2

    return TDMAsolver(Da, Db, Dc, rhs)


@nb.njit()
def gradient(array, h):
    grad_x = np.zeros_like(array)
    grad_y = np.zeros_like(array)

    nrows, ncols = array.shape

    for i in range(nrows):
        grad_x[i, :] = compact_scheme_6th(array[i, :], h)

    for i in range(ncols):
        grad_y[:, i] = compact_scheme_6th(array[:, i], h)

    return grad_x, grad_y
