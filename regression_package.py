import numpy as np


def simplest_design_matrix(x_vec):
    return np.array([np.ones(len(x_vec)), x_vec]).T


def hat_matrix(design_matrix):
    X = design_matrix
    return X @ np.linalg.inv(X.T @ X) @ X.T


def calc_influence(design_matrix):
    H = hat_matrix(design_matrix)
    return np.array([np.sqrt(1 - H[i, i]) for i in range(len(H))])


def std_estimator(residuals, dof):
    # dof = degrees of freedom
    std_square = (1 / (len(residuals) - dof)) * np.sum(np.square(residuals))
    return np.sqrt(std_square)


def var_prediction(design_matrix, sigma, x):
    return (sigma ** 2) * x.T @ np.linalg.inv(design_matrix.T @ design_matrix) @ x
