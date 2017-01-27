import os
from os.path import dirname, join, realpath
import numpy as np
from scipy.stats import gaussian_kde, norm, cauchy
from pystan import StanModel

__location__ = realpath(join(os.getcwd(), dirname(__file__)))

def obrien_fleming(information_fraction, alpha=0.05):
	"""
	Calculate an approximation of the O'Brien-Fleming alpha spending function.

	Args:

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

	Returns:
		boolean
	"""
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

	mu_x = np.mean(x)
	mu_y = np.mean(y)
	sigma_x = np.std(x)
	sigma_y = np.std(y)
	n_x = len(x)
	n_y = len(y)
	z = (mu_y-mu_x) / np.sqrt(sigma_x**2/n_x+sigma_y**2/n_y)
    
	if z > bound or z < -bound:
		return True
	else:
		return False


def bayes_factor(x, y, distribution='normal'):
	"""
	Args:
		sm (pystan.model.StanModel): precompiled Stan model object
		simulation_index (int): random seed used for the simulation
		day_index (int): time step of the peeking
		kpi (str): KPI name

	Returns:
		boolean
	"""
	if distribution == 'normal':
		fit_data = {'Nc': len(x), 
					'Nt': len(y), 
					'x': x, 
					'y': y}
	else:
		raise NotImplementedError
	model_file = __location__ + '/../models/' + distribution + '_kpi.stan'
	sm = StanModel(file=model_file)

	fit = sm.sampling(data=fit_data, iter=25000, chains=4, n_jobs=1)
	traces = fit.extract()
	kde = gaussian_kde(traces['delta'])

	prior = cauchy.pdf(0, loc=0, scale=1)
	bf = kde.evaluate(0)[0] / prior

	return bf > 3 or bf < 1/3.


if __name__ == '__main__':
	res = obrien_fleming(np.linspace(0,1,5+1)[1:])