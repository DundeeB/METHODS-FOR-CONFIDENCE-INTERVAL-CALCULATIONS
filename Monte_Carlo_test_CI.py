from working_example_global_parameters import *
from scipy.stats import norm
from scipy.special import betaincinv
import pymc as pm
from arviz import hdi


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


def Bayesian_Binomial_CI(n, k, CL=0.95):
    # Given the constant prior p(q)=1, and given k = np.sum(B) and n=len(B), we have:
    # p(q|B)=(n+1)(n choose k)q^k(1-q)^(n-k)
    # And so we need to solve for q_d, q_u, s.t. int from q_d to q_u of p(q|B) = 0.95. This define our 95CI. The result
    # is expressed using the betaincinv function
    # q_from_x = lambda x: betaincinv(k + 1, n - k + 1, x / (n + 1))
    q_from_x = lambda x: betaincinv(k + 1, n - k + 1, x)
    return q_from_x((1 - CL) / 2), q_from_x(CL + (1 - CL) / 2)


def MC_test_Bayesian_Binomial_CI(n, true_q, realizations=int(1e4)):
    """
    :param n: length of Binomial sample
    :param true_q: the parameter of the Binomial distribution
    :param realizations: number of realizations used in the monte carlo calculation
    :return: the CL, calculated from the MC realizations.
    """
    counter = 0
    for real in range(realizations):
        CI = Bayesian_Binomial_CI(n, np.random.binomial(n, true_q))
        if CI[0] <= true_q <= CI[1]:
            counter += 1
    return float(counter) / realizations


def Bayesian_CI_for_linear_regression(x_vec, sample, x_prediction, CL):
    mu_a, mu_b, sig_a, sig_b = 0, 0, 10, 10
    with pm.Model() as model:
        a = pm.Normal("a", mu=mu_a, sigma=sig_a)
        b = pm.Normal("b", mu=mu_b, sigma=sig_b)
        pm.Deterministic("prediction", a * x_prediction + b)
        pm.Normal("obs", mu=a * x_vec + b, sigma=sigma, observed=sample)
        linear_fit = pm.sample()
    return np.array(hdi(linear_fit, hdi_prob=CL, var_names="prediction").to_array())[0]


if __name__ == "__main__":
    # print('Analytic solution; percentage of times range includes true value is ' + \
    #       str(np.round(MC_test_CI(analytic_solution_CI), 3)))

    # true_q = 0.75
    # n = 10000
    # print('Bayesian Binomial; percentage of times range includes true value is ' + \
    #       str(np.round(MC_test_Bayesian_Binomial_CI(n, true_q), 3)))

    print('Bayesian credible interval; percentage of times interval includes true value is ' + \
          str(np.round(MC_test_CI(Bayesian_CI_for_linear_regression, realizations=10, CL=.95), 3)))
