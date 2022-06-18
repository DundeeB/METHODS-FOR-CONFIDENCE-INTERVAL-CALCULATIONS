import numpy as np
from regression_package import *

y_true = lambda x: 2 * x + 3
x_vec = np.array([i for i in range(1, 31)])
sigma = 10
x_prediction = 40

# np.random.normal(0, scale=sigma, size=len(x_vec))
epsilon_sampled = np.array([1.5696514, 9.70721346, -8.98851826, 16.80501892,
                            21.49349844, 1.70102863, -8.49491201, 10.56493773,
                            17.0855639, 1.91983716, 9.89007291, -8.68422734,
                            3.83159999, -1.46182969, -14.33304104, 1.31011658,
                            1.1925366, 1.27476029, 0.4110668, 7.13089034,
                            1.8995182, 6.15586736, -0.67727801, 8.41899763,
                            -13.39880163, 9.55726545, -3.14007926, -2.10523179,
                            0.19798023, -11.58942361])

best_fit_params = np.polyfit(x_vec, y_true(x_vec) + epsilon_sampled, 1)
sigma_prediction = np.sqrt(var_prediction(simplest_design_matrix(x_vec), sigma, np.array([1, x_prediction])))
