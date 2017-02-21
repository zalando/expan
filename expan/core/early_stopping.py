import os
from os.path import dirname, join, realpath
import numpy as np
from scipy.stats import gaussian_kde, norm, cauchy
from pystan import StanModel

import expan.core.statistics as statx

__location__ = realpath(join(os.getcwd(), dirname(__file__)))

def obrien_fleming(information_fraction, alpha=0.05):
	"""
	Calculate an approximation of the O'Brien-Fleming alpha spending function.

	Args:
		information_fraction: share of the information amount at the point 
			of evaluation, e.g. the share of the maximum sample size
		alpha: type-I error rate

	Returns:
		float: redistributed alpha value at the time point with the given 
			   information fraction
	"""
	return (1-norm.cdf(norm.ppf(1-alpha/2)/np.sqrt(information_fraction)))*2


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
		tuple: 
			- stop label
			- effect size (delta)
			- confidence interval of delta
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

	# if scalar, assume equal spacing between the intervals
	#if not isinstance(information_fraction, list): 
	#	fraction = np.linspace(0,1,information_fraction+1)[1:]
	#else:
	#	fraction = information_fraction

	# alpha spending function
	if spending_function in ('obrien_fleming'):
		func = eval(spending_function)
	else:
		raise NotImplementedError
	alpha_new = func(information_fraction, alpha=alpha)

	# calculate the z-score bound
	bound = norm.ppf(1-alpha_new/2)
	# replace potential inf with an upper bound
	if bound == np.inf:
		bound = cap

	mu_x = np.nanmean(_x)
	mu_y = np.nanmean(_y)
	sigma_x = np.nanstd(_x)
	sigma_y = np.nanstd(_y)
	n_x = statx.sample_size(_x)
	n_y = statx.sample_size(_y)
	z = (mu_x-mu_y) / np.sqrt(sigma_x**2/n_x+sigma_y**2/n_y)
    
	if z > bound or z < -bound:
		stop = 1
	else:
		stop = 0

	interval = statx.normal_difference(mu_x,sigma_x,n_x,mu_y,sigma_y,n_y,[alpha_new*100/2,100-alpha_new*100/2])

	return stop, mu_x-mu_y, interval, n_x, n_y, mu_x, mu_y


def HDI_from_MCMC(posterior_samples, credible_mass=0.95):
    # Computes highest density interval from a sample of representative values,
    # estimated as the shortest credible interval
    # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
    # http://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0]*nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]

    #ciWidth = np.zaeros(nCIs)
    #ciWidth[:] = sorted_points[ciIdxInc:] - sorted_points

    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
    return(HDImin, HDImax)


def _bayes_sampling(x, y, distribution='normal'):
	"""
	Helper function.

	Args:
		x (array_like): sample of a treatment group
		y (array_like): sample of a control group
		distribution: name of the KPI distribution model, which assumes a
			Stan model file with the same name exists

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
					'x': int(_x), 
					'y': int(_y)}
	else:
		raise NotImplementedError
	model_file = __location__ + '/../models/' + distribution + '_kpi.stan'
	sm = StanModel(file=model_file)

	fit = sm.sampling(data=fit_data, iter=25000, chains=4, n_jobs=1, seed=1)
	traces = fit.extract()

	return traces, n_x, n_y, mu_x, mu_y


def bayes_factor(x, y, distribution='normal'):
	"""
	Args:
		x (array_like): sample of a treatment group
		y (array_like): sample of a control group
		distribution: name of the KPI distribution model, which assumes a
			Stan model file with the same name exists

	Returns:
		tuple: 
			- stop label
			- effect size (delta)
			- credible interval of delta
			- sample size of x
			- sample size of y
			- absolute mean of x
			- absolute mean of y
	"""
	traces, n_x, n_y, mu_x, mu_y = _bayes_sampling(x, y, distribution=distribution)
	kde = gaussian_kde(traces['delta'])

	prior = cauchy.pdf(0, loc=0, scale=1)
	bf = kde.evaluate(0)[0] / prior
	stop = int(bf > 3 or bf < 1/3.)

	interval = HDI_from_MCMC(traces['alpha'])

	return stop, mu_x-mu_y, {'lower':interval[0],'upper':interval[1]}, n_x, n_y, mu_x, mu_y


def bayes_precision(x, y, distribution='normal', posterior_width=0.08):
	"""
	Args:
		x (array_like): sample of a treatment group
		y (array_like): sample of a control group
		distribution: name of the KPI distribution model, which assumes a
			Stan model file with the same name exists
		posterior_width: the stopping criterion, threshold of the posterior 
			width

	Returns:
		tuple: 
			- stop label
			- effect size (delta)
			- credible interval of delta
			- sample size of x
			- sample size of y
			- absolute mean of x
			- absolute mean of y
	"""
	traces, n_x, n_y, mu_x, mu_y = _bayes_sampling(x, y, distribution=distribution)
	interval = HDI_from_MCMC(traces['delta'])
	stop = int(interval[1] - interval[0] < posterior_width)

	return stop, mu_x-mu_y, {'lower':interval[0],'upper':interval[1]}, n_x, n_y, mu_x, mu_y


if __name__ == '__main__':
	#res = obrien_fleming(np.linspace(0,1,5+1)[1:])
	#res = obrien_fleming(0.5)
	np.random.seed(0)
	rand_s1 = np.random.normal(loc=0, size=1000)
	rand_s2 = np.random.normal(loc=0.1, size=1000)
	stop,delta,interval,n_x,n_y,mu_x,mu_y = bayes_precision(rand_s1, rand_s2)