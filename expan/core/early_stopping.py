import os
import logging
import pickle
import sys
import tempfile
from os.path import dirname, join, realpath

import pandas as pd
import numpy as np
from pystan import StanModel
from scipy.stats import gaussian_kde, norm, cauchy

import expan.core.statistics as statx
from expan.core.util import drop_nan
from expan.core.results import BaseTestStatistics, SampleStatistics, EarlyStoppingTestStatistics

__location__ = realpath(join(os.getcwd(), dirname(__file__)))
logger = logging.getLogger(__name__)

cache_sampling_results = False
sampling_results = {}  # memorized sampling results


def obrien_fleming(information_fraction, alpha=0.05):
    """ Calculate an approximation of the O'Brien-Fleming alpha spending function.

    :param information_fraction: share of the information  amount at the point of evaluation, 
                                 e.g. the share of the maximum sample size
    :type  information_fraction: float
    :param alpha: type-I error rate
    :type  alpha: float

    :return: redistributed alpha value at the time point with the given information fraction
    :rtype:  float
    """
    return (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(information_fraction))) * 2


def make_group_sequential(spending_function='obrien_fleming', estimated_sample_size=None, alpha=0.05, cap=8):
    """ A closure to the group_sequential function. """
    def go(x, y, x_denominators=1, y_denominators=1):

        # these next too lines are wrong, but they are bug-compatible with v0.6.13 !
        x = x / np.nanmean(x_denominators)
        y = y / np.nanmean(y_denominators)

        return group_sequential(x, y, spending_function, estimated_sample_size, alpha, cap)
    return go


def group_sequential(x, y, spending_function='obrien_fleming', estimated_sample_size=None, alpha=0.05, cap=8):
    """ Group sequential method to determine whether to stop early.

    :param x: sample of a treatment group
    :type  x: pd.Series or array-like
    :param y: sample of a control group
    :type  y: pd.Series or array-like
    :param spending_function: name of the alpha spending function, currently supports only 'obrien_fleming'.
    :type  spending_function: str
    :param estimated_sample_size: sample size to be achieved towards the end of experiment
    :type  estimated_sample_size: int
    :param alpha: type-I error rate
    :type  alpha: float
    :param cap: upper bound of the adapted z-score
    :type  cap: int

    :return: results of type EarlyStoppingTestStatistics
    :rtype:  EarlyStoppingTestStatistics
    """
    # Checking if data was provided and it has correct format
    if x is None or y is None:
        raise ValueError('Please provide two non-empty samples.')
    if not isinstance(x, pd.Series) and not isinstance(x, np.ndarray) and not isinstance(x, list):
        raise TypeError('Please provide samples of type Series or list.')
    if type(x) != type(y):
        raise TypeError('Please provide samples of the same type.')

    logger.info("Started running group sequential early stopping; spending function is {}, size of treatment is {} "
                "and size of control is {}".format(spending_function, len(x), len(y)))

    # Coercing missing values to right format
    _x = np.array(x, dtype=float)
    _y = np.array(y, dtype=float)

    n_x = statx.sample_size(_x)
    n_y = statx.sample_size(_y)

    if not estimated_sample_size:
        information_fraction = 1.0
    else:
        information_fraction = min(1.0, (n_x + n_y) / estimated_sample_size)

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

    treatment_statistics = SampleStatistics(int(n_x), float(np.nanmean(_x)), float(np.nanvar(_x)))
    control_statistics   = SampleStatistics(int(n_y), float(np.nanmean(_y)), float(np.nanvar(_y)))
    variant_statistics   = BaseTestStatistics(control_statistics, treatment_statistics)
    p_value              = statx.compute_p_value_from_samples(_x, _y)
    statistical_power    = statx.compute_statistical_power_from_samples(_x, _y, alpha)

    logger.info("Finished running group sequential early stopping; spending function is {}, size of treatment is {} "
                "and size of control is {}".format(spending_function, len(x), len(y)))
    return EarlyStoppingTestStatistics(variant_statistics.control_statistics,
                                       variant_statistics.treatment_statistics,
                                       float(mu_x - mu_y), interval, p_value, statistical_power, stop)


def HDI_from_MCMC(posterior_samples, credible_mass=0.95):
    """ Computes highest density interval from a sample of representative values, 
    estimated as the shortest credible interval.
    Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95).
    http://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    
    :param posterior_samples: sample of data points from posterior distribution of some parameter
    :type  posterior_samples: array-like
    :param credible_mass: the range of credible interval. 0.95 means 95% represents credible interval.
    :type  credible_mass: float

    :return: corresponding lower and upper bound for the credible interval
    :rtype:  tuple[float]
    """

    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0] * nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]

    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth)) + ciIdxInc]
    return (HDImin, HDImax)


def _bayes_sampling(x, y, distribution='normal', num_iters=25000, inference="sampling"):
    """ Helper function for bayesian sampling.

    :param x: sample of a treatment group
    :type  x: pd.Series or list (array-like)
    :param y: sample of a control group
    :type  y: pd.Series or list (array-like)
    :param distribution: name of the KPI distribution model, which assumes a Stan model file with the same name exists
    :type  distribution: str
    :param num_iters: number of iterations of sampling
    :type  num_iters: int
    :param inference: 'sampling' for MCMC sampling method or 'variational' for variational inference
    :type  inference: str

    :return: the posterior samples, sample size of x, sample size of y, absolute mean of x, absolute mean of y
    :rtype:  tuple[array-like, array-like, array-like, float, float]
    """
    # Checking if data was provided and it has correct format
    if x is None or y is None:
        raise ValueError('Please provide two non-empty samples.')
    if not isinstance(x, pd.Series) and not isinstance(x, np.ndarray) and not isinstance(x, list):
        raise TypeError('Please provide samples of type Series or list.')
    if type(x) != type(y):
        raise TypeError('Please provide samples of the same type.')

    logger.info("Started running bayesian inference with {} procedure, treatment group of size {}, "
                "control group of size {}, {} distribution.".format(inference, len(x), len(y), distribution, inference))

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

    logger.info("Finished running bayesian inference with {} procedure, treatment group of size {}, "
                "control group of size {}, {} distribution.".format(inference, len(x), len(y), distribution))
    return traces, n_x, n_y, mu_x, mu_y


def make_bayes_factor(distribution='normal', num_iters=25000, inference='sampling'):
    """ Closure method for the bayes_factor"""
    def f(x, y, x_denominators = 1, y_denominators = 1):
        x = x / np.nanmean(x_denominators)
        y = y / np.nanmean(y_denominators)
        return bayes_factor(x, y, distribution, num_iters, inference)
    return f


def bayes_factor(x, y, distribution='normal', num_iters=25000, inference='sampling'):
    """ Bayes factor computation.

    :param x: sample of a treatment group
    :type  x: pd.Series or list (array-like)
    :param y: sample of a control group
    :type  y: pd.Series or list (array-like)
    :param distribution: name of the KPI distribution model, which assumes a Stan model file with the same name exists
    :type  distribution: str
    :param num_iters: number of iterations of bayes sampling
    :type  num_iters: int
    :param inference: sampling or variational inference method for approximation the posterior
    :type  inference: str

    :return: results of type EarlyStoppingTestStatistics (without p-value and stat. power)
    :rtype:  EarlyStoppingTestStatistics
    """

    logger.info("Started running bayes factor with {} procedure, treatment group of size {}, "
                "control group of size {}, {} distribution.".format(len(x), len(y), distribution, inference))

    traces, n_x, n_y, mu_x, mu_y = _bayes_sampling(x, y, distribution=distribution, num_iters=num_iters,
                                                   inference=inference)
    trace_normalized_effect_size = get_trace_normalized_effect_size(distribution, traces)
    trace_absolute_effect_size = traces['delta']

    kde = gaussian_kde(trace_normalized_effect_size)
    prior = cauchy.pdf(0, loc=0, scale=1)
    # BF_01
    bf = kde.evaluate(0)[0] / prior
    stop = bf > 3 or bf < 1 / 3.

    credible_mass = 0.95
    left_out      = 1.0 - credible_mass
    p1            = round(left_out/2.0, 5)
    p2            = round(1.0 - left_out/2.0, 5)
    credible_interval = HDI_from_MCMC(trace_absolute_effect_size, credible_mass)

    treatment_statistics = SampleStatistics(int(n_x), float(mu_x), float(np.nanvar(x)))
    control_statistics   = SampleStatistics(int(n_y), float(mu_y), float(np.nanvar(y)))
    variant_statistics   = BaseTestStatistics(control_statistics, treatment_statistics)

    logger.info("Finished running bayes factor with {} procedure, treatment group of size {}, "
                "control group of size {}, {} distribution.".format(len(x), len(y), distribution, inference))

    return EarlyStoppingTestStatistics(variant_statistics.control_statistics,
                                       variant_statistics.treatment_statistics,
                                       float(mu_x - mu_y),
                                       dict([(p*100, v) for p, v in zip([p1, p2], credible_interval)]),
                                       None, None, stop)


def make_bayes_precision(distribution='normal', posterior_width=0.08, num_iters=25000, inference='sampling'):
    """ Closure method for the bayes_precision"""
    def f(x, y, x_denominators = 1, y_denominators = 1):
        x = x / np.nanmean(x_denominators)
        y = y / np.nanmean(y_denominators)
        return bayes_precision(x, y, distribution, posterior_width, num_iters, inference)
    return f


def bayes_precision(x, y, distribution='normal', posterior_width=0.08, num_iters=25000, inference='sampling'):
    """ Bayes precision computation.

    :param x: sample of a treatment group
    :type  x: pd.Series or list (array-like)
    :param y: sample of a control group
    :type  y: pd.Series or list (array-like)
    :param distribution: name of the KPI distribution model, which assumes a Stan model file with the same name exists
    :type  distribution: str
    :param posterior_width: the stopping criterion, threshold of the posterior  width
    :type  posterior_width: float
    :param num_iters: number of iterations of bayes sampling
    :type  num_iters: int
    :param inference: sampling or variational inference method for approximation the posterior
    :type  inference: str

    :return: results of type EarlyStoppingTestStatistics (without p-value and stat. power)
    :rtype:  EarlyStoppingTestStatistics
    """

    logger.info("Started running bayes precision with {} procedure, treatment group of size {}, "
                "control group of size {}, {} distribution.".format(len(x), len(y), distribution, inference))

    traces, n_x, n_y, mu_x, mu_y = _bayes_sampling(x, y, distribution=distribution,
                                                   num_iters=num_iters, inference=inference)
    trace_normalized_effect_size = get_trace_normalized_effect_size(distribution, traces)
    trace_absolute_effect_size = traces['delta']

    credible_mass = 0.95
    left_out      = 1.0 - credible_mass
    p1            = round(left_out/2.0, 5)
    p2            = round(1.0 - left_out/2.0, 5)

    credible_interval_delta            = HDI_from_MCMC(trace_absolute_effect_size, credible_mass)
    credible_interval_delta_normalized = HDI_from_MCMC(trace_normalized_effect_size, credible_mass)

    stop = credible_interval_delta_normalized[1] - credible_interval_delta_normalized[0] < posterior_width

    treatment_statistics = SampleStatistics(int(n_x), float(mu_x), float(np.nanvar(x)))
    control_statistics   = SampleStatistics(int(n_y), float(mu_y), float(np.nanvar(y)))
    variant_statistics   = BaseTestStatistics(control_statistics, treatment_statistics)

    logger.info("Finished running bayes precision with {} procedure, treatment group of size {}, "
                "control group of size {}, {} distribution.".format(len(x), len(y), distribution, inference))

    return EarlyStoppingTestStatistics(variant_statistics.control_statistics,
                                       variant_statistics.treatment_statistics,
                                       float(mu_x - mu_y),
                                       dict([(p * 100, v) for p, v in zip([p1, p2], credible_interval_delta)]),
                                       None, None, stop)


def get_trace_normalized_effect_size(distribution, traces):
    """ Obtaining a Stan model statistics for 'normal' or 'poisson' distribution

    :param distribution: name of the KPI distribution model, which assumes a Stan model file with the same name exists
    :type  distribution: str
    :param traces: sampling statistics
    :type  traces: dict

    :return: sample of data points from posterior distribution of some parameter
    :rtype:  array-like
    """
    # check of traces type, if traces is not dict "traces['alpha']" will return KeyError
    if not isinstance(traces, dict):
        raise TypeError("Traces statistics is not a dictionary and does not contain alpha or delta statistics.")

    if distribution == 'normal':
        return traces['alpha']
    elif distribution == 'poisson':
        variance = np.nanmean(np.array(traces['delta']))
        return traces['delta'] / np.sqrt(np.absolute(variance))
    else:
        raise ValueError("Model " + distribution + " is not implemented.")


def get_or_compile_stan_model(model_file, distribution):
    """ Creates Stan model. Compiles a Stan model and saves it to .pkl file to the folder selected by tempfile module if
    file doesn't exist yet and load precompiled model if there is a model file in temporary dir.
    
    Note: compiled_model_file is the hardcoded file path which may cause some issues in future.
    There are 2 alternative implementations for Stan models handling:

    1. Using global variables
    2. Pre-compiling stan models and adding them as a part of ExpAn project

    Using temporary files with tempfile module is not currently possible,
    since it generates a unique file name which is difficult to track.
    However, compiled modules are saved in temporary directory using tempfile module 
    which vary based on the current platform and settings. Cleaning up a temp dir is done on boot.

    :param model_file: model file location
    :type  model_file: str
    :param distribution: name of the KPI distribution model, which assumes a Stan model file with the same name exists
    :type  distribution: str

    :return: compiled Stan model for the selected distribution or normal distribution as a default option
    :rtype:  Class representing a compiled Stan model
    """

    logger.info("Started loading and compiling Stan model for {} distribution".format(distribution))

    if distribution is not 'normal' and distribution is not 'poisson':
        raise ValueError("Model " + distribution + " is not implemented.")

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
