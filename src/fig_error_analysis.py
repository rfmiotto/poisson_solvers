"""
This scripts create figures of the results for each mat file matching the specified
pattern. Here, I'm studying how noise affect the solution using different BCs and
solvers (CG, Jacobi and Jacobi with convolution).
For each method (solver + BC = number of items in `SOLUTION_FILE_PATTERNS`), 
2 figures are created:
    - A figure with table of results
    - A figure with table of residuals over iterations
"""

import fnmatch
import os
import re

import matplotlib.pyplot as plt
import scipy

SOLUTION_FILE_PATTERNS = {
    "cg_ground_truth_bc*.mat": "fig_cg_ground_truth_bc",
    "cg_zero_dirichlet_bc*.mat": "fig_cg_zero_dirichlet",
    "jacobi_ground_truth_bc*.mat": "fig_jacobi_ground_truth_bc",
    "jacobi_zero_dirichlet_bc*.mat": "fig_jacobi_zero_dirichlet",
    "jacobi_zero_neumann_bc*.mat": "fig_jacobi_zero_neumann",
    "conv_jacobi_ground_truth_bc*.mat": "fig_conv_jacobi_ground_truth_bc",
}


def main(pattern: str, out_prefix: str):
    solution_files = get_solution_files(pattern)

    dpi = 100
    nrows = 3
    ncols = 4
    height = (256 + 50) * nrows
    width = (256 + 50) * ncols
    fig1, axes1 = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        layout="compressed",
    )
    fig1.set_tight_layout({"pad": 0})

    fig2, axes2 = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        layout="compressed",
    )
    fig2.set_tight_layout({"pad": 0})

    for k, sol in enumerate(solution_files):

        _, solution, residuals = read_solution(sol)
        noise_lvl = get_noise_lvl_from_filename(sol)

        i, j = convert_to_ij(k)
        axes1[i, j].imshow(solution, cmap="gray")
        axes1[i, j].set_axis_off()
        axes1[i, j].set_title(f"noise lvl: {noise_lvl}")

        axes2[i, j].plot(residuals, color="k")
        axes2[i, j].set_yscale("log")
        axes2[i, j].set_title(f"noise lvl: {noise_lvl}")

    # remove axis from empty subplots
    for i in range(nrows):
        for j in range(ncols):
            axes1[i, j].set_axis_off()
    axes2[-1, -1].set_axis_off()

    fig1.savefig(f"{out_prefix}_results.png", transparent=False)
    fig2.savefig(f"{out_prefix}_residuals.png", transparent=False)


def get_noise_lvl_from_filename(filename: str) -> float:
    # pylint: disable=anomalous-backslash-in-string
    return float(re.findall("\d+\.\d+", filename)[0])


def convert_to_ij(k: int) -> tuple[int, int]:
    ncols = 4
    i = int(k / ncols)
    j = k % ncols
    return i, j


def get_solution_files(pattern: str) -> list[str]:
    files = []
    for f in os.listdir("."):
        if fnmatch.fnmatch(f, pattern):
            files.append(f)

    if not files:
        raise FileNotFoundError("No file was found")
    return sorted(files)


def read_solution(file: str):
    f = scipy.io.loadmat(file)
    source = f["source"]
    solution = f["solution"]
    residuals = f["residuals"].squeeze()
    # noise_lvl = f["noise_lvl"][0][0]
    return source, solution, residuals  # , noise_lvl


if __name__ == "__main__":
    for sol_pattern, prefix in SOLUTION_FILE_PATTERNS.items():
        main(sol_pattern, prefix)
