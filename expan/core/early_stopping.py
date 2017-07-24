import os
from os.path import dirname, join, realpath

import numpy as np
import sys
from pystan import StanModel
from scipy.stats import gaussian_kde, norm, cauchy
from expan.core.util import drop_nan

import expan.core.statistics as statx

import pickle
import tempfile

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


def make_group_sequential(spending_function='obrien_fleming', information_fraction=1.0, alpha=0.05, cap=8):
    def f(x, y):
        return group_sequential(x, y, spending_function, information_fraction,
                                alpha, cap)
    return f

def group_sequential(x,
                     y,
                     spending_function='obrien_fleming',
                     information_fraction=1,
                     alpha=0.05,
                     cap=8):
    """
    Group sequential method to determine whether to stop early or not.

    Args:
        x (array_like): sample of a treatment group
        y (array_like): sample of a control group
        spending_function: name of the alpha spending function, currently
            supports: 'obrien_fleming'
        information_fraction: share of the information amount at the point 
            of evaluation, e.g. the share of the maximum sample size
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

    # if scalar, assume equal spacing between the intervals
    # if not isinstance(information_fraction, list):
    #	fraction = np.linspace(0,1,information_fraction+1)[1:]
    # else:
    #	fraction = information_fraction

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
    n_x = statx.sample_size(_x)
    n_y = statx.sample_size(_y)
    z = (mu_x - mu_y) / np.sqrt(sigma_x ** 2 / n_x + sigma_y ** 2 / n_y)

    if z > bound or z < -bound:
        stop = True
    else:
        stop = False

    interval = statx.normal_difference(mu_x, sigma_x, n_x, mu_y, sigma_y, n_y,
                                       [alpha_new * 100 / 2, 100 - alpha_new * 100 / 2])

    # return stop, mu_x - mu_y, interval, n_x, n_y, mu_x, mu_y
    return {'stop'     : bool(stop),
            'delta'    : float(mu_x - mu_y),
            'interval' : interval,
            'n_x'      : int(n_x),
            'n_y'      : int(n_y),
            'mu_x'     : float(mu_x),
            'mu_y'     : float(mu_y)}


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

def _bayes_sampling(x, y, distribution='normal', num_iters=25000):
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

    key = (str(_x), str(_y), num_iters)

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

    fit = sm.sampling(data=fit_data, iter=num_iters, chains=4, n_jobs=1, seed=1,
                      control={'stepsize': 0.01, 'adapt_delta': 0.99})
    traces = fit.extract()

    if cache_sampling_results:
        sampling_results[key] = (traces, n_x, n_y, mu_x, mu_y)

    return traces, n_x, n_y, mu_x, mu_y


def make_bayes_factor(distribution='normal', num_iters=25000):
    def f(x, y):
        return bayes_factor(x, y, distribution, num_iters)
    return f


def bayes_factor(x, y, distribution='normal', num_iters=25000):
    """
    Args:
        x (array_like): sample of a treatment group
        y (array_like): sample of a control group
        distribution: name of the KPI distribution model, which assumes a
            Stan model file with the same name exists
        num_iters: number of iterations of bayes sampling

    Returns:
        dictionary with statistics
    """
    traces, n_x, n_y, mu_x, mu_y = _bayes_sampling(x, y, distribution=distribution, num_iters=num_iters)
    kde = gaussian_kde(traces['delta'])

    prior = cauchy.pdf(0, loc=0, scale=1)
    # BF_01
    bf = kde.evaluate(0)[0] / prior
    # stop = int(bf > 3 or bf < 1 / 3.)
    stop = bf > 3 or bf < 1 / 3.

    credibleMass = 0.95                # another magic number
    leftOut      = 1.0 - credibleMass
    p1           = round(leftOut/2.0, 5)
    p2           = round(1.0 - leftOut/2.0, 5)
    interval = HDI_from_MCMC(traces['delta'], credibleMass)

    # return stop, mu_x - mu_y, {'lower': interval[0], 'upper': interval[1]}, n_x, n_y, mu_x, mu_y
    return {'stop'     : bool(stop),
            'delta'    : float(mu_x - mu_y),
            'interval' : {p1*100: interval[0], p2*100: interval[1]},
            'n_x'      : int(n_x),
            'n_y'      : int(n_y),
            'mu_x'     : float(mu_x),
            'mu_y'     : float(mu_y),
            'num_iters': num_iters}


def make_bayes_precision(distribution='normal', posterior_width=0.08, num_iters=25000):
    def f(x, y):
        return bayes_precision(x, y, distribution, posterior_width, num_iters)
    return f

def bayes_precision(x, y, distribution='normal', posterior_width=0.08, num_iters=25000):
    """
    Args:
        x (array_like): sample of a treatment group
        y (array_like): sample of a control group
        distribution: name of the KPI distribution model, which assumes a
            Stan model file with the same name exists
        posterior_width: the stopping criterion, threshold of the posterior 
            width
        num_iters: number of iterations of bayes sampling

    Returns:
        dictionary with statistics
    """
    traces, n_x, n_y, mu_x, mu_y = _bayes_sampling(x, y, distribution=distribution, num_iters=num_iters)
    credibleMass = 0.95                # another magic number
    leftOut      = 1.0 - credibleMass
    p1           = round(leftOut/2.0, 5)
    p2           = round(1.0 - leftOut/2.0, 5)
    interval = HDI_from_MCMC(traces['delta'], credibleMass)

    # stop = int(interval[1] - interval[0] < posterior_width)
    stop = interval[1] - interval[0] < posterior_width

    # return stop, mu_x - mu_y, {'lower': interval[0], 'upper': interval[1]}, n_x, n_y, mu_x, mu_y
    return {'stop'     : bool(stop),
            'delta'    : float(mu_x - mu_y),
            'interval' : {p1*100: interval[0], p2*100: interval[1]},
            'n_x'      : int(n_x),
            'n_y'      : int(n_y),
            'mu_x'     : float(mu_x),
            'mu_y'     : float(mu_y),
            'num_iters': num_iters}
