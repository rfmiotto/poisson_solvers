"""
Solve Poisson using Jacobi's method with convolution (works well for 3D cases)
"""

import fnmatch
import os
from typing import Optional

import h5py
import numba as nb
import numpy as np
import scipy
from numba_progress import ProgressBar
from numpy.typing import NDArray

from src.laplacian import laplacian

h5f = h5py.File("../jhtdb/jhtdb_isotropic1024fine_3D_pressure.h5", "r")
xcoor = np.array(h5f["xcoor"])

SIZE = 256
NUM_ITERATIONS = 1_000_000
FILENAME = "jhtdb.mat"
SPACING = xcoor[1] - xcoor[0]


def main():
    source_files = get_list_of_source_files()

    for source_file in source_files:
        source, noise_lvl = get_source_term(source_file)

        u = np.zeros((SIZE, SIZE))
        u = set_dirichlet_bc(u, FILENAME)

        with ProgressBar(total=NUM_ITERATIONS) as progress:
            u, residuals = iterate(
                NUM_ITERATIONS,
                u,
                source,
                progress,
            )

        filename = f"jacobi_ground_truth_bc_{noise_lvl:.2f}_noise.mat"
        save_results(filename, u, residuals, source)


@nb.njit(nogil=True)
def iterate(num_iter, arr, source, progress, tol=1e-3):
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


def save_results(
    filename: str, solution: NDArray, residuals: list[float], source: NDArray
) -> None:
    mat_dict = {
        "solution": solution.reshape((SIZE, SIZE)),
        "residuals": residuals,
        "source": source.reshape((SIZE, SIZE)),
    }
    scipy.io.savemat(filename, mat_dict)


def get_list_of_source_files() -> list[str]:
    files = []
    for f in os.listdir("."):
        if fnmatch.fnmatch(f, "jhtdb_with_*_noise.mat"):
            files.append(f)

    if not files:
        raise FileNotFoundError("No file was found")
    return sorted(files)


def get_source_term(file: str) -> NDArray:
    f = scipy.io.loadmat(file)
    source = f["laplacian"]
    noise_lvl = f["noise_lvl"][0][0]
    return source, noise_lvl


def get_ground_truth(file: str) -> NDArray:
    return scipy.io.loadmat(file)["representation"]


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
