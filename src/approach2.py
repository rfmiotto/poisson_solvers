"""
Solve Poisson using Jacobi's method with convolution (works well for 3D cases)
"""

from functools import partial
import scipy
from scipy.ndimage import convolve, generate_binary_structure
import numpy as np
import numba as nb
from numba_progress import ProgressBar
import matplotlib.pyplot as plt
import h5py


f = h5py.File("../jhtdb/jhtdb_isotropic1024fine_3D_pressure.h5", "r")
xcoor = np.array(f["xcoor"])

SIZE = 256
NUM_ITERATIONS = 10_000
FILENAME = "jhtdb.mat"
SPACING = (xcoor[1] - xcoor[0]) / 2


def main():
    u = np.zeros((SIZE, SIZE))

    dirichlet_fn = partial(set_boundary_conditions, file=FILENAME)
    # u = set_boundary_conditions(u, FILENAME)

    source = get_source_term(FILENAME)
    source *= SPACING**2

    # source = add_noise(source)

    with ProgressBar(total=NUM_ITERATIONS) as progress:
        u = iterate(
            NUM_ITERATIONS, u, source, kernel_operator(), dirichlet_fn, progress
        )

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


def iterate(num_iter, arr, source, kernel, dirichlet_fn, progress):
    for _ in range(num_iter):
        arr = convolve(arr, kernel, mode="constant") - source
        arr = dirichlet_fn(arr)
        # arr = apply_neumann_bc(arr)

        progress.update(1)

    return arr


@nb.njit(nogil=True)
def apply_neumann_bc(u):
    u[:, 0] = u[:, 1]
    u[0, :] = u[1, :]
    u[:, -1] = u[:, -2]
    u[-1, :] = u[-2, :]
    return u


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
