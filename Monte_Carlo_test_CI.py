from working_example_global_parameters import *
from scipy.stats import norm


def MC_test_CI(CI_func_from_sample, realizations=int(1e5), CL=95.4 / 100, y_true=y_true, x_vec=x_vec,
               x_prediction=x_prediction):
    """
    Generate random noise realizations around the true straigt line. For each noise realization run the
    CI_func_from_sample to get the CI range, and finally count how many times the true value is inside the CI
    :param CI_func_from_sample: function which take in its first & second enrties the samples x1...xN, y1...yN, in its
     third entry x_prediction and in its fourth entry the CL (0<CL<1). It returns a tuple/list of lower and upper bounds
    :param realizations: int, # realizations for Monte Carlo calculation
    :return:
    """
    counter = 0
    for real in range(realizations):
        realized_noise = np.random.normal(0, scale=sigma, size=len(x_vec))
        sample = y_true(x_vec) + realized_noise
        CI_range = CI_func_from_sample(x_vec, sample, x_prediction, CL)
        if CI_range[0] < y_true(x_prediction) < CI_range[1]:
            counter += 1
    return float(counter) / realizations


def analytic_solution_CI(x_vec, sample, x_prediction, CL):
    fit = np.polyfit(x_vec, sample, 1)
    prediction = np.polyval(fit, x_prediction)
    num_of_sigmas_from_CL = norm.ppf(1 - (1 - CL) / 2)
    return [prediction - num_of_sigmas_from_CL * sigma_prediction,
            prediction + num_of_sigmas_from_CL * sigma_prediction]


if __name__ == "__main__":
    print('Analytic solution; percentage of times range includes true value is ' + \
          str(np.round(MC_test_CI(analytic_solution_CI), 3)))
