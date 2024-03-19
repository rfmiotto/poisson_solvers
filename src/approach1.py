"""
Solve Poisson using Jacobi's method with loops
"""

import h5py
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy
from numba_progress import ProgressBar

f = h5py.File("../jhtdb/jhtdb_isotropic1024fine_3D_pressure.h5", "r")
xcoor = np.array(f["xcoor"])

SIZE = 256
NUM_ITERATIONS = 1_000_000
FILENAME = "jhtdb.mat"
SPACING = xcoor[1] - xcoor[0]


def main():
    u = np.zeros((SIZE, SIZE))

    u = set_boundary_conditions(u, FILENAME)

    source = get_source_term(FILENAME)

    # source = add_noise(source)

    with ProgressBar(total=NUM_ITERATIONS) as progress:
        u, residuals = iterate(NUM_ITERATIONS, u, source, progress)

    plt.subplot(2, 1, 1)
    plt.plot(residuals)
    plt.yscale("log")
    plt.show()

    # u_wrong_bc = np.zeros((SIZE, SIZE))
    # u_wrong_bc = set_zero_dirichlet_boundary_condition(u_wrong_bc)

    # with ProgressBar(total=NUM_ITERATIONS) as progress:
    #     u_wrong_bc = iterate(NUM_ITERATIONS, u_wrong_bc, source, progress)

    ground_truth = scipy.io.loadmat(FILENAME)["representation"]

    vmax = ground_truth.max()
    vmin = ground_truth.min()

    _, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(ground_truth, vmax=vmax, vmin=vmin, cmap="gray")
    ax[1].imshow(u, vmax=vmax, vmin=vmin, cmap="gray")
    # ax[2].imshow(u_wrong_bc, vmax=vmax, vmin=vmin, cmap="gray")
    ax[0].set_title("Ground truth")
    ax[1].set_title("Poisson with correct BC")
    # ax[2].set_title("Poisson with zero BC")
    plt.show()


@nb.njit(nogil=True)
def iterate(num_iter: int, arr, source, progress):

    residuals = []

    scaled_source = source * SPACING**2

    num_y, num_x = arr.shape

    for iteration in range(num_iter):
        # arr = apply_neumann_bc(arr)
        for i in range(1, num_y - 1):
            for j in range(1, num_x - 1):
                arr[i, j] = 0.25 * (
                    arr[i + 1, j]
                    + arr[i - 1, j]
                    + arr[i, j + 1]
                    + arr[i, j - 1]
                    - scaled_source[i, j]
                )

        if iteration % 100 == 0:
            relative_residual_norm = np.linalg.norm(
                source - laplacian(arr, SPACING)
            ) / np.linalg.norm(4 * source)

            residuals.append(relative_residual_norm)

            if relative_residual_norm < 1e-5:
                return arr, residuals

        progress.update(1)

    return arr, residuals


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


@nb.njit()
def laplacian(arr, h):
    dsdx, dsdy = gradient(arr, h)
    part1, _ = gradient(dsdx, h)
    _, part2 = gradient(dsdy, h)
    return part1 + part2


@nb.njit(nogil=True)
def apply_neumann_bc(u):
    u[:, 0] = u[:, 1]
    u[0, :] = u[1, :]
    u[:, -1] = u[:, -2]
    u[-1, :] = u[-2, :]
    return u


def add_noise(source, factor: float = 0.01):
    source_amplitude = max(source.max(), source.min())
    noise_scale = factor * source_amplitude
    noise = noise_scale * np.random.normal(0, 1, size=SIZE**2).reshape((SIZE, SIZE))
    source += noise
    return source


def get_source_term(file: str):
    return scipy.io.loadmat(file)["laplacian"]


def set_boundary_conditions(u, file: str):
    true_field = scipy.io.loadmat(file)["representation"]

    u[:, 0] = true_field[:, 0]
    u[0, :] = true_field[0, :]
    u[:, -1] = true_field[:, -1]
    u[-1, :] = true_field[-1, :]

    return u


def set_zero_dirichlet_boundary_condition(u):
    u[:, 0] = 0.0
    u[0, :] = 0.0
    u[:, -1] = 0.0
    u[-1, :] = 0.0
    return u


if __name__ == "__main__":
    main()
