"""
Solve Poisson using conjugate gradient method
"""

from typing import TypedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray

f = h5py.File("../jhtdb/jhtdb_isotropic1024fine_3D_pressure.h5", "r")
xcoor = np.array(f["xcoor"])

SIZE = 256 - 2
FILENAME = "jhtdb.mat"
SPACING = xcoor[1] - xcoor[0]
NOISE_LEVELS = np.linspace(0, 0.20, 21)


# @nb.njit(nogil=True)
def laplacian_operator(u_vec):
    num_points = len(u_vec)
    n = int(np.sqrt(num_points))
    u_mat = np.reshape(u_vec, (n, n))

    laplacian_op = 4 * u_mat
    laplacian_op[:-1, :] -= u_mat[1:, :]  # -u(i+1, j)
    laplacian_op[1:, :] -= u_mat[:-1, :]  # -u(i-1, j)
    laplacian_op[:, :-1] -= u_mat[:, 1:]  # -u(i, j+1)
    laplacian_op[:, 1:] -= u_mat[:, :-1]  # -u(i, j-1)

    return laplacian_op.ravel()


class BoundaryConditions(TypedDict):
    boundary_condition1: NDArray
    boundary_condition2: NDArray
    boundary_condition3: NDArray
    boundary_condition4: NDArray


def load(source, bc: BoundaryConditions, spacing: float):
    b = -(spacing**2) * source.copy()

    b[:, 0] += bc["boundary_condition1"]
    b[0, :] += bc["boundary_condition2"]
    b[:, -1] += bc["boundary_condition3"]
    b[-1, :] += bc["boundary_condition4"]

    return b.ravel()


def main():
    num_points_inner = SIZE**2

    source = get_source_term(FILENAME)
    source_inner = source[1:-1, 1:-1]

    true_field = get_ground_truth(FILENAME)
    true_field_inner = true_field[1:-1, 1:-1]

    bc: BoundaryConditions
    bc = {
        "boundary_condition1": np.zeros(SIZE),  # using 0 as BC
        "boundary_condition2": np.zeros(SIZE),  # using 0 as BC
        "boundary_condition3": np.zeros(SIZE),  # using 0 as BC
        "boundary_condition4": np.zeros(SIZE),  # using 0 as BC
        # "boundary_condition1": true_field_inner[:, 0],  # using ground truth
        # "boundary_condition2": true_field_inner[0, :],  # using ground truth
        # "boundary_condition3": true_field_inner[:, -1],  # using ground truth
        # "boundary_condition4": true_field_inner[-1, :],  # using ground truth
    }

    def report(sol_vec):
        relative_residual_norm = np.linalg.norm(b - matrix @ sol_vec) / np.linalg.norm(
            b
        )
        residuals.append(relative_residual_norm)

    for noise_lvl in NOISE_LEVELS:
        source_noise = add_noise(source_inner, noise_lvl)

        b = load(source_noise, bc, SPACING)

        matrix = scipy.sparse.linalg.LinearOperator(
            (num_points_inner, num_points_inner), matvec=laplacian_operator
        )

        residuals = []

        u_inner, cg_info = scipy.sparse.linalg.cg(
            matrix,
            b,
            x0=0.5 * np.ones(num_points_inner),
            rtol="1e-5",
            maxiter=2_000,
            callback=report,
        )

        if cg_info != 0:
            print("Convergence to tolerance not achieved")

        filename = f"cg_zero_dirichlet_bc_{noise_lvl:.2f}_noise.mat"
        # filename = f"cg_ground_truth_bc_{noise_lvl:.2f}_noise.mat"
        save_results(filename, u_inner, residuals, source_noise)


def save_results(
    filename: str, solution: NDArray, residuals: list[float], source: NDArray
) -> None:
    mat_dict = {
        "solution": solution.reshape((SIZE, SIZE)),
        "residuals": residuals,
        "source": source.reshape((SIZE, SIZE)),
    }
    scipy.io.savemat(filename, mat_dict)


def add_noise(source: NDArray, factor: float = 0.01) -> NDArray:
    np.random.seed(0)
    source_amplitude = max(source.max(), source.min())
    noise_scale = factor * source_amplitude
    noise = noise_scale * np.random.normal(0, 1, size=SIZE**2).reshape((SIZE, SIZE))
    return source + noise


def get_source_term(file: str) -> NDArray:
    return scipy.io.loadmat(file)["laplacian"]


def get_ground_truth(file: str) -> NDArray:
    return scipy.io.loadmat(file)["representation"]


if __name__ == "__main__":
    main()
