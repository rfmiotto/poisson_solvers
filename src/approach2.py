"""
Solve Poisson using Jacobi's method with convolution (works well for 3D cases)
"""

from functools import partial
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy
from numba_progress import ProgressBar
from scipy.ndimage import convolve, generate_binary_structure

from src.laplacian import laplacian

f = h5py.File("../jhtdb/jhtdb_isotropic1024fine_3D_pressure.h5", "r")
xcoor = np.array(f["xcoor"])

SIZE = 256
NUM_ITERATIONS = 10_000
FILENAME = "jhtdb.mat"
SPACING = xcoor[1] - xcoor[0]


def main():
    u = np.zeros((SIZE, SIZE))

    boundary_condition_fn = partial(set_dirichlet_bc, filename=FILENAME)
    # boundary_condition_fn = partial(set_dirichlet_bc) # zero dirichlet
    # boundary_condition_fn = partial(set_neumann_bc)

    source = get_source_term(FILENAME)
    source *= SPACING**2

    # source = add_noise(source)

    with ProgressBar(total=NUM_ITERATIONS) as progress:
        u, residuals = iterate(
            NUM_ITERATIONS,
            u,
            source,
            kernel_operator(),
            boundary_condition_fn,
            progress,
        )

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


def iterate(num_iter, arr, source, kernel, bc_fn, progress, tol=1e-5):
    residuals = []

    scaled_source = 0.25 * source
    for iteration in range(num_iter):
        arr = convolve(arr, kernel, mode="constant") - scaled_source
        arr = bc_fn(arr)

        if iteration % 100 == 0:
            relative_residual_norm = np.linalg.norm(
                source - laplacian(arr, 1)
            ) / np.linalg.norm(source)

            residuals.append(relative_residual_norm)

            if relative_residual_norm < tol:
                return arr, residuals

        progress.update(1)

    return arr, residuals


# def laplacian(arr):
#     dsdx, dsdy = np.gradient(arr, edge_order=2)
#     part1 = np.gradient(dsdx, edge_order=2, axis=0)
#     part2 = np.gradient(dsdy, edge_order=2, axis=1)
#     return part1 + part2


def kernel_operator():
    kernel = generate_binary_structure(rank=2, connectivity=1).astype(float) / 4
    kernel[1, 1] = 0
    return kernel


def add_noise(source):
    source_amplitude = max(source.max(), source.min())
    noise_scale = 0.01 * source_amplitude
    noise = noise_scale * np.random.normal(0, 1, size=SIZE**2).reshape((SIZE, SIZE))
    source += noise
    return source


def get_source_term(file: str):
    return scipy.io.loadmat(file)["laplacian"]


def set_dirichlet_bc(u, filename: Optional[str] = ""):
    if filename:
        true_field = scipy.io.loadmat(filename)["representation"]

        u[:, 0] = true_field[:, 0]
        u[0, :] = true_field[0, :]
        u[:, -1] = true_field[:, -1]
        u[-1, :] = true_field[-1, :]
    else:
        u[:, 0] = 0.0
        u[0, :] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = 0.0
    return u


@nb.njit(nogil=True)
def set_neumann_bc(u):
    u[:, 0] = u[:, 1]
    u[0, :] = u[1, :]
    u[:, -1] = u[:, -2]
    u[-1, :] = u[-2, :]
    return u


if __name__ == "__main__":
    main()
