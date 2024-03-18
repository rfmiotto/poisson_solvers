"""
Solve Poisson using conjugate gradient method
"""

from typing import TypedDict

import h5py
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy
from numpy.typing import NDArray

f = h5py.File("../jhtdb/jhtdb_isotropic1024fine_3D_pressure.h5", "r")
xcoor = np.array(f["xcoor"])

SIZE = 256 - 2
FILENAME = "jhtdb.mat"
SPACING = xcoor[1] - xcoor[0]


def conjgrad(operator, b, x):

    r = b - operator(x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    for _ in range(len(b)):
        Ap = operator(p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


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

    return b.ravel()  # * spacing**2


def main():
    num_points_inner = SIZE**2

    source = get_source_term(FILENAME)
    source_inner = source[1:-1, 1:-1]

    matrix = scipy.sparse.linalg.LinearOperator(
        (num_points_inner, num_points_inner), matvec=laplacian_operator
    )

    true_field = scipy.io.loadmat(FILENAME)["representation"]
    true_field_inner = true_field[1:-1, 1:-1]
    bc: BoundaryConditions
    bc = {
        "boundary_condition1": true_field_inner[:, 0],
        "boundary_condition2": true_field_inner[0, :],
        "boundary_condition3": true_field_inner[:, -1],
        "boundary_condition4": true_field_inner[-1, :],
    }

    b = load(source_inner, bc, SPACING)

    # u_inner = conjgrad(laplacian_operator, b, np.zeros((num_points_inner)))

    u_inner, cg_info = scipy.sparse.linalg.cg(matrix, b, rtol="1e-12")

    if cg_info != 0:
        raise RuntimeWarning("Convergence to tolerance not achieved")

    ground_truth = scipy.io.loadmat(FILENAME)["representation"]
    ground_truth_inner = ground_truth[1:-1, 1:-1]

    vmax = ground_truth_inner.max()
    vmin = ground_truth_inner.min()

    _, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(ground_truth_inner, vmax=vmax, vmin=vmin, cmap="gray")
    # ax[1].imshow(u_inner.reshape((SIZE, SIZE)), vmax=vmax, vmin=vmin, cmap="gray")
    ax[1].imshow(u_inner.reshape((SIZE, SIZE)), cmap="gray")
    # ax[2].imshow(u_wrong_bc, vmax=vmax, vmin=vmin, cmap="gray")
    ax[0].set_title("Ground truth")
    ax[1].set_title("Poisson with correct BC")
    # ax[2].set_title("Poisson with zero BC")
    plt.show()

    # x = np.linspace(0, length, SIZE)
    # y = np.linspace(0, length, SIZE)
    # (xg, yg) = np.meshgrid(x, y)
    # u_mat = np.reshape(u, (SIZE, SIZE))
    # ax = plt.axes(projection="3d")
    # ax.plot_surface(xg, yg, u_mat)
    # plt.show()


def get_source_term(file: str):
    return scipy.io.loadmat(file)["laplacian"]


if __name__ == "__main__":
    main()
