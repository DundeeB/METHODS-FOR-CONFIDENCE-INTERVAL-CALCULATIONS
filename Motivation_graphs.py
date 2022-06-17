import numpy as np
from matplotlib import pyplot as plt

default_plt_kwargs = {'linewidth': 3, 'markersize': 20}
size = 30
params = {'legend.fontsize': size * 0.75, 'figure.figsize': (14, 8), 'axes.labelsize': size, 'axes.titlesize': size,
          'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75}
plt.rcParams.update(params)
y_true = lambda x: 2 * x + 3
x_vec = np.array([i for i in range(1, 31)])
sigma = 10
x_prediction = 40


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


def sampled_noise():
    # np.random.normal(0, scale=sigma, size=len(x_vec))
    epsilon_sampled = np.array([1.5696514, 9.70721346, -8.98851826, 16.80501892,
                                21.49349844, 1.70102863, -8.49491201, 10.56493773,
                                17.0855639, 1.91983716, 9.89007291, -8.68422734,
                                3.83159999, -1.46182969, -14.33304104, 1.31011658,
                                1.1925366, 1.27476029, 0.4110668, 7.13089034,
                                1.8995182, 6.15586736, -0.67727801, 8.41899763,
                                -13.39880163, 9.55726545, -3.14007926, -2.10523179,
                                0.19798023, -11.58942361])
    return epsilon_sampled


def plot_line_noise_and_best_fit(x_vec, x_vec_sampled, y_true, y_with_noise, y_hat):
    plt.plot(x_vec, y_true(x_vec), label='True straight line', c='b', **default_plt_kwargs)
    plt.plot(x_vec_sampled, y_with_noise, '.', label='Sampled data, $\\sigma=10$', c='g', zorder=-1,
             **default_plt_kwargs)
    plt.plot(x_vec, y_hat, label='Best fit to sampled data', c='r', **default_plt_kwargs)


def demonstrate_straight_line_noise_and_best_fit():
    epsilon_sampled = sampled_noise()
    y_with_noise = y_true(x_vec) + epsilon_sampled
    y_hat = hat_matrix(simplest_design_matrix(x_vec)) @ y_with_noise.T

    plt.figure()
    plt.grid()
    plot_line_noise_and_best_fit(x_vec, x_vec, y_true, y_with_noise, y_hat)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Graphs/Straight_line_noise_and_best_fit')


def demonstrate_prediction():
    epsilon_sampled = sampled_noise()
    best_fit_params = np.polyfit(x_vec, y_true(x_vec) + epsilon_sampled, 1)
    x_vec_w_predicted_x = np.concatenate((x_vec, [x_prediction]))
    y_hat_w_prediction = np.polyval(best_fit_params, x_vec_w_predicted_x)
    plt.figure()
    plot_line_noise_and_best_fit(x_vec_w_predicted_x, x_vec, y_true, y_true(x_vec) + epsilon_sampled,
                                 y_hat_w_prediction)
    err_keywargs = {'capsize': 10, 'capthick': 3, 'elinewidth': 3, 'linewidth': 3}
    sig_from_noise = std_estimator(residuals=y_true(x_vec) + epsilon_sampled - np.polyval(best_fit_params, x_vec),
                                   dof=2)
    plt.errorbar(x_prediction, y_hat_w_prediction[-1], yerr=2 * sig_from_noise, c='r',
                 label='Extrapulated value & range $\\pm 2 \\hat{\\sigma}$', **err_keywargs)
    plt.savefig('Graphs/prediction_for_cover')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.savefig('Graphs/Line_noise_best_fit_and_prediction')


if __name__ == "__main__":
    demonstrate_straight_line_noise_and_best_fit()
    demonstrate_prediction()
    plt.show()
