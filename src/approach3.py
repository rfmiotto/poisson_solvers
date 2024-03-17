"""
Solve Poisson using conjugate gradient method
"""

import scipy
from scipy.sparse.csgraph import laplacian
from scipy.ndimage import generate_binary_structure
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import h5py

f = h5py.File("../jhtdb/jhtdb_isotropic1024fine_3D_pressure.h5", "r")
xcoor = np.array(f["xcoor"])

SIZE = 256
NUM_ITERATIONS = 10_000
FILENAME = "jhtdb.mat"
SPACING = (xcoor[1] - xcoor[0]) / 2


def laplacian_operator(u_vec):
    # apply the discrete negative Laplacian with homogeneous Dirichlet boundary conditions

    num_points = len(u_vec)
    n = int(np.sqrt(num_points))
    u_mat = np.reshape(u_vec, (n, n))

    laplacian_op = 4 * u_mat

    laplacian_op[:-1, :] -= u_mat[1:, :]
    laplacian_op[1:, :] -= u_mat[:-1, :]
    laplacian_op[:, 1:] -= u_mat[:, :-1]
    laplacian_op[:, :-1] -= u_mat[:, 1:]

    h2 = (n + 1) * (n + 1)  # FIXME: I think this should be the spacing**2
    laplacian_op *= h2
    laplacian_op = np.reshape(laplacian_op, (num_points))  # TODO: ravel() ??

    return laplacian_op


def load(source, f1, f2, f3, f4):
    # construct load vector for system -\Delta u = f
    n = len(f1)
    b = source.copy()
    h2 = (n + 1) * (n + 1)  # FIXME: This should be the spacing**2

    # y=0
    b[0, :] += f1 * h2

    # x=0
    b[:, 0] += f2 * h2

    # y=length
    b[-1, :] += f3 * h2

    # x=length
    b[:, -1] += f4 * h2

    b = np.reshape(b, (n**2))  # TODO: ravel() ??
    return b


def main_web():
    size = 100  # number of side points (I think it is the inner region, not considering the edges)
    length = 1
    h = length / (size + 1)  # spacing
    num_points = size**2

    x = np.linspace(h, length - h, size)
    y = np.linspace(h, length - h, size)
    xv = np.reshape(x, (1, size))
    yv = np.reshape(y, (size, 1))

    # source
    sig = 0.1
    source = (
        100
        * np.exp((-((yv - 0.5) ** 2)) / sig**2)
        * np.exp(-((xv - 0.5) ** 2) / sig**2)
        / (2 * np.pi * sig)
    )

    # y=0
    f1 = 1 * np.sin(x * np.pi)

    # x=0
    f2 = 2 * np.sin(y * np.pi)

    # y=length
    f3 = 3 * np.sin(x * np.pi)

    # x=length
    f4 = 4 * np.sin(y * np.pi)

    matrix_a = scipy.sparse.linalg.LinearOperator(
        (num_points, num_points), matvec=laplacian_operator
    )
    b = load(source, f1, f2, f3, f4)

    u, exitcode = scipy.sparse.linalg.cg(matrix_a, b)

    print(exitcode)

    (xg, yg) = np.meshgrid(x, y)
    u_mat = np.reshape(u, (size, size))
    ax = plt.axes(projection="3d")
    ax.plot_surface(xg, yg, u_mat)
    plt.show()


def main():
    u = np.zeros((SIZE, SIZE))

    u = set_boundary_conditions(u, FILENAME)

    source = get_source_term(FILENAME)
    source *= SPACING**2

    # source = add_noise(source)

    operator = laplacian(kernel_operator(), form="lo")
    linear_operator = scipy.sparse.linalg.LinearOperator((3, 3), matvec=operator)
    u = scipy.sparse.linalg.cg(linear_operator, source)

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


def kernel_operator():
    kernel = generate_binary_structure(rank=2, connectivity=1).astype(float) / 4
    kernel[1, 1] = 0
    return kernel


@nb.njit(nogil=True)
def apply_neumann_bc(u):
    u[:, 0] = u[:, 1]
    u[0, :] = u[1, :]
    u[:, -1] = u[:, -2]
    u[-1, :] = u[-2, :]
    return u


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
    main_web()
