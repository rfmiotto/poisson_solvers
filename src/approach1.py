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
NUM_ITERATIONS = 10_000
FILENAME = "jhtdb.mat"
SPACING = xcoor[1] - xcoor[0]


def main():
    u = np.zeros((SIZE, SIZE))

    u = set_boundary_conditions(u, FILENAME)

    source = get_source_term(FILENAME)
    source *= SPACING**2

    # source = add_noise(source)

    with ProgressBar(total=NUM_ITERATIONS) as progress:
        u = iterate(NUM_ITERATIONS, u, source, progress)

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
def iterate(num_iter: int, arr, scaled_source, progress):
    num_y, num_x = arr.shape
    for _ in range(num_iter):
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

        progress.update(1)

    return arr


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
