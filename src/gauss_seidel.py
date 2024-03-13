import numpy as np


def gauss_seidel(matrix_A, vector_b, tolerance, max_iterations, solution_vector):

    for iteration in range(max_iterations):
        print(f"The solution vector in iteration {iteration} is: {solution_vector}")

        x_old = solution_vector.copy()

        for i in range(matrix_A.shape[0]):
            solution_vector[i] = (
                vector_b[i]
                - np.dot(matrix_A[i, :i], solution_vector[:i])
                - np.dot(matrix_A[i, (i + 1) :], x_old[(i + 1) :])
            ) / matrix_A[i, i]

        l_norm_inf = max(abs((solution_vector - x_old))) / max(abs(x_old))
        print(f"The L infinity norm in iteration {iteration} is: {l_norm_inf}")
        if l_norm_inf < tolerance:
            break

    return solution_vector
