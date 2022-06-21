from matplotlib import pyplot as plt
from working_example_global_parameters import *
from scipy.special import binom

default_plt_kwargs = {'linewidth': 3, 'markersize': 20}
size = 30
params = {'legend.fontsize': size * 0.75, 'figure.figsize': (14, 8), 'axes.labelsize': size, 'axes.titlesize': size,
          'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75}
plt.rcParams.update(params)


def plot_true_line_sample_and_best_fit(x_vec, x_vec_sampled, y_true, y_with_noise, y_hat):
    plt.plot(x_vec, y_true(x_vec), label='True straight line', c='b', **default_plt_kwargs)
    plt.plot(x_vec_sampled, y_with_noise, '.', label='Sampled data, $\\sigma=10$', c='g', zorder=-1,
             **default_plt_kwargs)
    plt.plot(x_vec, y_hat, label='Best fit to sampled data', c='r', **default_plt_kwargs)


def wrap_line_sample_and_best_fit_for_figure():
    y_with_noise = y_true(x_vec) + epsilon_sampled
    y_hat = hat_matrix(simplest_design_matrix(x_vec)) @ y_with_noise.T

    plt.figure()
    plt.grid()
    plot_true_line_sample_and_best_fit(x_vec, x_vec, y_true, y_with_noise, y_hat)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Graphs/Straight_line_noise_and_best_fit')


def demonstrate_prediction(range_size=4 * sigma):
    x_vec_w_predicted_x = np.concatenate((x_vec, [x_prediction]))
    y_hat_w_prediction = np.polyval(best_fit_params, x_vec_w_predicted_x)
    plt.figure()
    plot_true_line_sample_and_best_fit(x_vec_w_predicted_x, x_vec, y_true, y_true(x_vec) + epsilon_sampled,
                                       y_hat_w_prediction)
    err_keywargs = {'capsize': 10, 'capthick': 3, 'elinewidth': 3, 'linewidth': 3}
    plt.errorbar(x_prediction, y_hat_w_prediction[-1], yerr=range_size / 2, c='r',
                 label='Extrapulated value & range', **err_keywargs)
    plt.axis('off')
    plt.savefig('Graphs/prediction_for_cover')

    plt.axis('on')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.savefig('Graphs/Line_noise_best_fit_and_prediction')


def plot_0954_CI():
    # Plots \pm 2 \sigma CI, which is 0.954 CI (95.4%)
    demonstrate_prediction(range_size=4 * sigma_prediction)
    x_range = 3.5 * (x_prediction - np.mean(x_vec))
    x_for_plotting_CI = np.linspace(np.mean(x_vec) - x_range / 2, np.mean(x_vec) + x_range / 2, 2 * len(x_vec))
    sig_x = np.array(
        [np.sqrt(var_prediction(simplest_design_matrix(x_vec), sigma, np.array([1, x]))) for x in x_for_plotting_CI])

    y_predicted = np.polyval(best_fit_params, x_for_plotting_CI)
    plt.plot(x_for_plotting_CI, y_predicted + 2 * sig_x, '--r', label='95.4%CI', **default_plt_kwargs)
    plt.plot(x_for_plotting_CI, y_predicted - 2 * sig_x, '--r', **default_plt_kwargs)
    plt.plot(x_for_plotting_CI, y_true(x_for_plotting_CI), '-b', **default_plt_kwargs)
    plt.plot(x_for_plotting_CI, y_predicted, '-r', **default_plt_kwargs)
    plt.legend()
    plt.savefig('Graphs/Analytic_CI')
    return


def plot_binomial_Bayes_belief(prior_sample):
    """
    Shows the progress in the posterior of sampling from binomial distributation, when the prior is constant over [0,1]
    :param prior_sample:
    :return:
    """
    q = np.linspace(0, 1)
    plt.figure()
    plt.plot(q, 1 + 0 * q, label='prior', **default_plt_kwargs)
    for partial_sample_length in range(1, len(prior_sample) + 1):
        partial_sample = prior_sample[:partial_sample_length]
        n = len(partial_sample)
        k = np.sum(partial_sample)
        p_q_given_B = (n + 1) * binom(n, k) * (q ** k) * ((1 - q) ** (n - k))
        line_style = '-' if n % 2 == 0 else '--'
        plt.plot(q, p_q_given_B, line_style, label=str(partial_sample), **default_plt_kwargs)
    plt.xlabel('q')
    plt.ylabel('p(q|sample)')
    plt.legend()
    plt.savefig('Graphs/Bayes_Binomial_propagation')


if __name__ == "__main__":
    # wrap_line_sample_and_best_fit_for_figure()
    # print(sigma_prediction)
    # demonstrate_prediction(range_size=4 * sigma_prediction)
    # plot_0954_CI()
    plot_binomial_Bayes_belief([0, 1, 0, 1, 0, 1])
    plt.show()
