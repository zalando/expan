# fileencoding: utf8
import os
import unittest
import warnings

import numpy as np
import pandas as pd

import expan.core.results as r
from expan.core.experiment import Experiment
from tests.tests_core.test_data import generate_random_data
from expan.core.results import prob_uplift_over_zero_single_metric
import imp

imp.reload(r)

data_dir = os.getcwd() + '/tests/tests_core/'  # TODO: adjust depending on where we're called from?


def generate_random_results():
	# TODO
	pass


def load_example_results():
	"""
	This just loads example data so that we need always generate random stuff
	in order to test.

	Also demonstrates loading of HDF5 into expan

	Returns Results object.
	"""

	example_fname = 'example_results.h5'
	example_fpath = os.path.join(data_dir, example_fname)

	return r.from_hdf(example_fpath)

class ResultsTestCase(unittest.TestCase):
	"""
	Defines the setUp() and tearDown() functions for the results test cases.
	"""

	def setUp(self):
		"""
	    Load the needed datasets for all TestCases and set the random
	    seed so that randomized algorithms show deterministic behaviour.
	    """
		np.random.seed(0)
		self.data = Experiment('B', *generate_random_data())
		# Create time column. TODO: Do this nicer
		self.data.kpis['time_since_treatment'] = \
			self.data.features['treatment_start_time']
		# Make time part of index
		self.data.kpis.set_index('time_since_treatment', append=True, inplace=True)

	def tearDown(self):
		"""
	    Clean up after the test
	    """
		# TODO: find out if we have to remove data manually
		pass


class ResultsClassTestCase(ResultsTestCase):
	def testExampleResults(self):
		h5py_available = False
		import imp
		try:
			imp.find_module('h5py')
			imp.find_module('tables')
			h5py_available = True
		except Exception:
			warnings.warn(
				"""Could not import h5py or tables module. Skipping
		        testExampleResults(). Please make sure that you have the h5py
		        and tables packages installed."""
			)

		if h5py_available:
			#aa = load_example_results()
			warnings.warn("No data for h5 loading available... skipping tests of example h5 data")

	def test_relative_uplift_delta(self):
		"""Check if the calculation of relative uplift for delta results is
	    correct.
	    """
		res = self.data.delta()
		df = res.relative_uplift('delta', 'normal_same')
		np.testing.assert_almost_equal(df, np.array([[-4.219601, 0]]), decimal=5)

	def test_prob_uplift_over_zero_single_metric(self):
		"""Check if the conversion from confidence intervals to probability is correct for one metric."""
		res = self.data.delta(kpi_subset=['normal_same'])
		#df = prob_uplift_over_zero_single_metric(res.df, self.data.baseline_variant)
		np.testing.assert_almost_equal(res.df.loc[pd.IndexSlice[:,:,:,'prob_uplift_over_0'], 'value'],
									   np.array([[0.946519, np.nan]]), decimal=5)

	def test_prob_uplift_over_zero_multiple_metric(self):
		"""Check if the conversion from confidence intervals to probability is correct for multiple metrics."""
		res = self.data.delta(kpi_subset=['normal_same','normal_shifted'])
		#res.calculate_prob_uplift_over_zero()
		np.testing.assert_almost_equal(res.df.loc[pd.IndexSlice[:,:,:,'prob_uplift_over_0'], 'value'],
									   np.array([[0.946519,np.nan],[0,np.nan]]), decimal=5)

if __name__ == '__main__':
	#unittest.main()
	np.random.seed(0)
	exp = Experiment('B', *generate_random_data())
	res = exp.delta(['normal_shifted'])
