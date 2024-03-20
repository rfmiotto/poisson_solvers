"""
Solve Poisson using conjugate gradient method
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray

SIZE = 256
FILENAME = "jhtdb.mat"
NOISE_LEVELS = np.linspace(0, 0.1, 11)


def main():
    np.random.seed(0)  # reproducible noise

    source = get_source_term(FILENAME)
    source_inner = source[1:-1, 1:-1]

    dpi = 100
    nrows = 3
    ncols = 4
    height = (256 + 50) * nrows
    width = (256 + 50) * ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        layout="compressed",
    )
    fig.set_tight_layout({"pad": 0})

    for k, noise_lvl in enumerate(NOISE_LEVELS):
        source[1:-1, 1:-1] = add_noise(source_inner, noise_lvl)

        filename = f"jhtdb_with_{noise_lvl:.2f}_noise.mat"

        i, j = convert_to_ij(k)

        axes[i, j].imshow(source, cmap="gray")
        axes[i, j].set_axis_off()
        axes[i, j].set_title(f"noise lvl: {noise_lvl}")

        save_results(filename, source, noise_lvl)

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].set_axis_off()
    plt.savefig("source_noise_levels.png", transparent=False)


def convert_to_ij(k: int) -> tuple[int, int]:
    ncols = 4
    i = int(k / ncols)
    j = k % ncols
    return i, j


def save_results(filename: str, source: NDArray, noise_lvl) -> None:
    mat_dict = {
        "laplacian": source.reshape((SIZE, SIZE)),
        "noise_lvl": noise_lvl,
    }
    scipy.io.savemat(filename, mat_dict)


def add_noise(source: NDArray, factor: float = 0.01) -> NDArray:
    nrows, ncols = source.shape
    size = nrows * ncols

    source_amplitude = max(source.max(), source.min())
    noise_scale = factor * source_amplitude

    noise = noise_scale * np.random.normal(0, 1, size=size).reshape(source.shape)

    return source + noise


def get_source_term(file: str) -> NDArray:
    return scipy.io.loadmat(file)["laplacian"]


if __name__ == "__main__":
    main()
