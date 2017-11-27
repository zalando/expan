import os
import pickle
import sys
import tempfile
from os.path import dirname, join, realpath

import numpy as np
from pystan import StanModel
from scipy.stats import gaussian_kde, norm, cauchy

import expan.core.statistics as statx
from expan.core.util import drop_nan

__location__ = realpath(join(os.getcwd(), dirname(__file__)))


def obrien_fleming(information_fraction, alpha=0.05):
    """
    Calculate an approximation of the O'Brien-Fleming alpha spending function.

    Args:
        information_fraction (scalar or array_like): share of the information 
            amount at the point of evaluation, e.g. the share of the maximum 
            sample size
        alpha: type-I error rate

    Returns:
        float: redistributed alpha value at the time point with the given 
               information fraction
    """
    return (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(information_fraction))) * 2


def make_group_sequential(spending_function='obrien_fleming', estimated_sample_size=None, alpha=0.05, cap=8):
    def f(x, y):
        return group_sequential(x, y, spending_function, estimated_sample_size,
                                alpha, cap)
    return f


def group_sequential(x,
                     y,
                     spending_function='obrien_fleming',
                     estimated_sample_size=None,
                     alpha=0.05,
                     cap=8):
    """
    Group sequential method to determine whether to stop early or not.

    Args:
        x (array_like): sample of a treatment group
        y (array_like): sample of a control group
        spending_function: name of the alpha spending function, currently
            supports: 'obrien_fleming'
        estimated_sample_size: sample size to be achieved towards
            the end of experiment
        alpha: type-I error rate
        cap: upper bound of the adapted z-score

    Returns:
        EarlyStoppingStatistics object
    """
    # Checking if data was provided
    if x is None or y is None:
        raise ValueError('Please provide two non-None samples.')

    # Coercing missing values to right format
    _x = np.array(x, dtype=float)
    _y = np.array(y, dtype=float)

    n_x = statx.sample_size(_x)
    n_y = statx.sample_size(_y)

    if not estimated_sample_size:
        information_fraction = 1.0
    else:
        information_fraction = min(1.0, min(n_x, n_y) / estimated_sample_size)

    # alpha spending function
    if spending_function in ('obrien_fleming'):
        func = eval(spending_function)
    else:
        raise NotImplementedError
    alpha_new = func(information_fraction, alpha=alpha)

    # calculate the z-score bound
    bound = norm.ppf(1 - alpha_new / 2)
    # replace potential inf with an upper bound
    if bound == np.inf:
        bound = cap

    mu_x = np.nanmean(_x)
    mu_y = np.nanmean(_y)
    sigma_x = np.nanstd(_x)
    sigma_y = np.nanstd(_y)
    z = (mu_x - mu_y) / np.sqrt(sigma_x ** 2 / n_x + sigma_y ** 2 / n_y)

    if z > bound or z < -bound:
        stop = True
    else:
        stop = False

    interval = statx.normal_difference(mu_x, sigma_x, n_x, mu_y, sigma_y, n_y,
                                       [alpha_new * 100 / 2, 100 - alpha_new * 100 / 2])

    # return stop, mu_x - mu_y, interval, n_x, n_y, mu_x, mu_y
    interval = [{'percentile': p, 'value': v} for (p, v) in interval.items()]
    return {'stop'                  : bool(stop),
            'delta'                 : float(mu_x - mu_y),
            'confidence_interval'   : interval,
            'treatment_sample_size' : int(n_x),
            'control_sample_size'   : int(n_y),
            'treatment_mean'        : float(mu_x),
            'control_mean'          : float(mu_y)}


def HDI_from_MCMC(posterior_samples, credible_mass=0.95):
    # Computes highest density interval from a sample of representative values,
    # estimated as the shortest credible interval
    # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
    # http://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0] * nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]

    # ciWidth = np.zaeros(nCIs)
    # ciWidth[:] = sorted_points[ciIdxInc:] - sorted_points

    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth)) + ciIdxInc]
    return (HDImin, HDImax)


def get_or_compile_stan_model(model_file, distribution):
    """
    Creates Stan model. Compiles a Stan model and saves it to .pkl file to the folder selected by tempfile module if
        file doesn't exist yet and load precompiled model if there is a model file in temporary dir.
    Args:
        model_file: model file location
        distribution: name of the KPI distribution model, which assumes a 
            Stan model file with the same name exists
    Returns:
        returns compiled Stan model for the selected distribution or normal distribution
            as a default option
    Note: compiled_model_file is the hardcoded file path which may cause some issues in future.
    There are 2 alternative implementations for Stan models handling:
        1. Using global variables
        2. Pre-compiling stan models and adding them as a part of expan project
        (3). Using temporary files with tempfile module is not currently possible, since it 
            generates a unique file name which is difficult to track.
        However, compiled modules are saved in temporary directory using tempfile module 
        which vary based on the current platform and settings. Cleaning up a temp dir is done on boot.
    """
    python_version = '{0[0]}.{0[1]}'.format(sys.version_info)
    compiled_model_file = tempfile.gettempdir() + '/expan_early_stop_compiled_stan_model_' \
                          + distribution + '_' + python_version + '.pkl'

    if os.path.isfile(compiled_model_file):
        sm = pickle.load(open(compiled_model_file, 'rb'))
    else:
        sm = StanModel(file=model_file)
        with open(compiled_model_file, 'wb') as f:
            pickle.dump(sm, f)
    return sm

cache_sampling_results = False
sampling_results = {} # memoized sampling results


def _bayes_sampling(x, y, distribution='normal', num_iters=25000, inference="sampling"):
    """
    Helper function.

    Args:
        x (array_like): sample of a treatment group
        y (array_like): sample of a control group
        distribution: name of the KPI distribution model, which assumes a
            Stan model file with the same name exists
        num_iters: number of iterations of sampling

    Returns:
        tuple:
            - the posterior samples
            - sample size of x
            - sample size of y
            - absolute mean of x
            - absolute mean of y
    """
    # Checking if data was provided
    if x is None or y is None:
        raise ValueError('Please provide two non-None samples.')

    # Coercing missing values to right format
    _x = np.array(x, dtype=float)
    _y = np.array(y, dtype=float)
    _x = drop_nan(_x)
    _y = drop_nan(_y)

    key = (str(_x), str(_y), num_iters, inference)

    if cache_sampling_results and key in sampling_results:
        return sampling_results[key]

    mu_x = np.nanmean(_x)
    mu_y = np.nanmean(_y)
    n_x = statx.sample_size(_x)
    n_y = statx.sample_size(_y)

    if distribution == 'normal':
        fit_data = {'Nc': n_y,
                    'Nt': n_x,
                    'x': _x,
                    'y': _y}
    elif distribution == 'poisson':
        fit_data = {'Nc': n_y,
                    'Nt': n_x,
                    'x': _x.astype(int),
                    'y': _y.astype(int)}
    else:
        raise NotImplementedError

    model_file = __location__ + '/../models/' + distribution + '_kpi.stan'

    sm = get_or_compile_stan_model(model_file, distribution)

    if inference == "sampling":
        fit = sm.sampling(data=fit_data, iter=num_iters, chains=4, n_jobs=1, seed=1,
                          control={'stepsize': 0.01, 'adapt_delta': 0.99})
        traces = fit.extract()

    elif inference == "variational":
        results_dict = sm.vb(data=fit_data, iter=10000)
        traces = {}
        for i in range(len(results_dict['sampler_param_names'])):
            para_name = results_dict['sampler_param_names'][i]
            para_values = np.array(results_dict['sampler_params'][i])
            traces[para_name] = para_values

    if cache_sampling_results:
        sampling_results[key] = (traces, n_x, n_y, mu_x, mu_y)

    return traces, n_x, n_y, mu_x, mu_y


def make_bayes_factor(distribution='normal', num_iters=25000, inference='sampling'):
    def f(x, y):
        return bayes_factor(x, y, distribution, num_iters, inference)
    return f


def bayes_factor(x, y, distribution='normal', num_iters=25000, inference='sampling'):
    """
    Args:
        x (array_like): sample of a treatment group
        y (array_like): sample of a control group
        distribution: name of the KPI distribution model, which assumes a
            Stan model file with the same name exists
        num_iters: number of iterations of bayes sampling
        inference: sampling or variational inference method for approximation the posterior

    Returns:
        dictionary with statistics
    """
    traces, n_x, n_y, mu_x, mu_y = _bayes_sampling(x, y, distribution=distribution, num_iters=num_iters,
                                                   inference=inference)
    trace_normalized_effect_size = get_trace_normalized_effect_size(distribution, traces)
    trace_absolute_effect_size = traces['delta']

    kde = gaussian_kde(trace_normalized_effect_size)
    prior = cauchy.pdf(0, loc=0, scale=1)
    # BF_01
    bf = kde.evaluate(0)[0] / prior
    stop = bf > 3 or bf < 1 / 3.

    credibleMass = 0.95                # another magic number
    leftOut      = 1.0 - credibleMass
    p1           = round(leftOut/2.0, 5)
    p2           = round(1.0 - leftOut/2.0, 5)
    credible_interval = HDI_from_MCMC(trace_absolute_effect_size, credibleMass)

    return {'stop'                  : bool(stop),
            'delta'                 : float(mu_x - mu_y),
            'confidence_interval'   : [{'percentile': p*100, 'value': v} for p, v in zip([p1, p2], credible_interval)],
            'treatment_sample_size' : int(n_x),
            'control_sample_size'   : int(n_y),
            'treatment_mean'        : float(mu_x),
            'control_mean'          : float(mu_y),
            'number_of_iterations'  : num_iters}


def get_trace_normalized_effect_size(distribution, traces):
    if distribution == 'normal':
        return traces['alpha']
    elif distribution == 'poisson':
        variance = np.nanmean(np.array(traces['delta']))
        return traces['delta'] / np.sqrt(np.absolute(variance))
    else:
        raise ValueError("model " + distribution + " is not implemented.")


def make_bayes_precision(distribution='normal', posterior_width=0.08, num_iters=25000, inference='sampling'):
    def f(x, y):
        return bayes_precision(x, y, distribution, posterior_width, num_iters, inference)
    return f


def bayes_precision(x, y, distribution='normal', posterior_width=0.08, num_iters=25000, inference='sampling'):
    """
    Args:
        x (array_like): sample of a treatment group
        y (array_like): sample of a control group
        distribution: name of the KPI distribution model, which assumes a
            Stan model file with the same name exists
        posterior_width: the stopping criterion, threshold of the posterior 
            width
        num_iters: number of iterations of bayes sampling
        inference: sampling or variational inference method for approximation the posterior

    Returns:
        dictionary with statistics
    """
    traces, n_x, n_y, mu_x, mu_y = _bayes_sampling(x, y, distribution=distribution, num_iters=num_iters,
                                                   inference=inference)
    trace_normalized_effect_size = get_trace_normalized_effect_size(distribution, traces)
    trace_absolute_effect_size = traces['delta']

    credibleMass = 0.95                # another magic number
    leftOut      = 1.0 - credibleMass
    p1           = round(leftOut/2.0, 5)
    p2           = round(1.0 - leftOut/2.0, 5)

    credible_interval_delta            = HDI_from_MCMC(trace_absolute_effect_size, credibleMass)
    credible_interval_delta_normalized = HDI_from_MCMC(trace_normalized_effect_size, credibleMass)

    stop = credible_interval_delta_normalized[1] - credible_interval_delta_normalized[0] < posterior_width

    return {'stop'                  : bool(stop),
            'delta'                 : float(mu_x - mu_y),
            'confidence_interval'   : [{'percentile': p*100, 'value': v} for p, v in zip([p1, p2], credible_interval_delta)],
            'treatment_sample_size' : int(n_x),
            'control_sample_size'   : int(n_y),
            'treatment_mean'        : float(mu_x),
            'control_mean'          : float(mu_y),
            'number_of_iterations'  : num_iters}
