import unittest
import warnings

import numpy as np
import pandas as pd

from expan.core.experiment import Experiment, subgroup_deltas, time_dependent_deltas
from tests.tests_core.test_data import generate_random_data

# raise the same warning multiple times
#warnings.simplefilter('always', UserWarning)

from expan.core.debugging import Dbg


class ExperimentTestCase(unittest.TestCase):
	"""
  Defines the setUp() and tearDown() functions for the statistics test cases.
  """

	def setUp(self):
		"""
    Load the needed datasets for all StatisticsTestCases and set the random
    seed so that randomized algorithms show deterministic behaviour.
    """
		np.random.seed(0)
		self.data = Experiment('B', *generate_random_data(), dbg=Dbg(dbg_lvl=5))
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


class ExperimentNonClassTestCases(ExperimentTestCase):
	"""
  Test cases for the non-class functions.
  """

	def test__subgroup_deltas__computation(self):
		"""
    Check if subgroup_deltas() functions properly
    """
		# Calculate result
		result = subgroup_deltas(self.data.metrics.reset_index()
								 [['variant', 'feature', 'normal_shifted']], variants=['A', 'B'])
		# check uplift
		df = result.statistic('sga', 'uplift', 'normal_shifted', 'feature')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-0.980599, -1.001506]), decimal=5)
		# check pctile
		df = result.statistic('sga', 'uplift_pctile', 'normal_shifted', 'feature')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-1.036771, -0.924426, -1.058535, -0.944478]), decimal=5)
		# check samplesize
		df = result.statistic('sga', 'sample_size', 'normal_shifted', 'feature')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.array([[3067, 1966], [3041, 1926]]), decimal=5)
		# check variant_mean
		df = result.statistic('sga', 'variant_mean', 'normal_shifted', 'feature')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.array([[0.001221, 0.981820], [-0.012310, 0.989196]]), decimal=5)

	def test__time_dependent_deltas__computation(self):
		"""
    Check if time_dependent_deltas() functions properly
    """
		# Calculate result
		result = time_dependent_deltas(self.data.metrics.reset_index()
									   [['variant', 'time_since_treatment', 'normal_shifted']], variants=['A', 'B'])
		# check uplift
		df = result.statistic('trend', 'uplift', 'normal_shifted')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-1.009421, -0.847400, -1.119885, -1.042597, -0.868819,
												 -1.091165, -0.952307, -1.028234, -0.978774, -0.985696]), decimal=5)
		# check pctile
		df = result.statistic('trend', 'uplift_pctile', 'normal_shifted')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-1.137482, -0.881360, -0.970678, -0.724122, -1.245795,
												 -0.993975, -1.178494, -0.906699, -0.993683, -0.743954, -1.225361,
												 -0.956969, -1.082180, -0.822435, -1.151715, -0.904753, -1.095209,
												 -0.862340, -1.109407, -0.861985]), decimal=5)
		# check samplesize
		df = result.statistic('trend', 'sample_size', 'normal_shifted')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.column_stack(([649, 595, 600, 590, 625, 602, 607, 608, 616, 616],
														[405, 401, 378, 362, 377, 369, 406, 392, 414, 388])), decimal=5)
		# check variant_mean
		df = result.statistic('trend', 'variant_mean', 'normal_shifted')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.column_stack(([0.005761, 0.057487, -0.067107, 0.001125, 0.093085,
														 -0.067894, -0.030500, -0.060996, 0.016257, -0.006091],
														[1.015182, 0.904887, 1.052778, 1.043721, 0.961904, 1.023271,
														 0.921807, 0.967238, 0.995031, 0.979605])), decimal=5)


class ExperimentClassTestCases(ExperimentTestCase):
	"""
  Test cases for the Experiment class functions.
  """

	def test__feature_check__computation(self):
		"""
    Check if feature check is corectly performed on test data set
    """
		# Perform feature check
		result = self.data.feature_check()
		# check delta
		df = result.statistic('fc', 'pre_treatment_diff', 'treatment_start_time')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-0.023670]), decimal=5)
		# check pctile
		df = result.statistic('fc', 'pre_treatment_diff_pctile', 'treatment_start_time')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-0.140031, 0.092690]), decimal=5)
		# check samplesize
		df = result.statistic('fc', 'sample_size', 'treatment_start_time')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.array([[6108, 3892]]), decimal=5)
		# check variant_mean
		df = result.statistic('fc', 'variant_mean', 'treatment_start_time')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.array([[4.493287, 4.516958]]), decimal=5)

	def test__sga__index_levels(self):
		"""
    Check if sga() returns the proper index levels
    """
		# Perform sga()
		result = self.data.sga()
		# Check if all index levels are present
		index_levels = [
			pd.Index([u'normal_same', u'normal_shifted', u'normal_shifted_by_feature', u'normal_unequal_variance'],
					 dtype='object', name=u'metric'),
			pd.Index([u'feature', u'treatment_start_time'], dtype='object', name=u'subgroup_metric'),
			pd.Index([u'[0.0,2.0)', u'[2.0,4.0)', u'[4.0,7.0)', u'[7.0,9.0]', u'{has}', u'{non}'], dtype='object',
					 name=u'subgroup'),
			pd.Index([u'sample_size', u'uplift', u'uplift_pctile', u'variant_mean'], dtype='object', name=u'statistic'),
			pd.Float64Index([2.5, 97.5], dtype='float64', name=u'pctile')
		]
		result_levels = list(result.df.index.levels)
		# Check if all index levels match expectation TODO: Make nice
		np.testing.assert_array_equal(index_levels[0], result_levels[0])
		np.testing.assert_array_equal(index_levels[1], result_levels[1])
		np.testing.assert_array_equal(index_levels[2], result_levels[2])
		np.testing.assert_array_equal(index_levels[3], result_levels[3])
		np.testing.assert_array_equal(index_levels[4], result_levels[4])

	def test__sga__computation(self):
		"""
    Check if sga() performs proper computation
    """
		# Perform sga()
		result = self.data.sga()
		# check uplift
		df = result.statistic('sga', 'uplift', 'normal_shifted', 'feature')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-0.980599, -1.001506]), decimal=5)
		# check pctile
		df = result.statistic('sga', 'uplift_pctile', 'normal_shifted', 'feature')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-1.036771, -0.924426, -1.058535, -0.944478]), decimal=5)
		# check samplesize
		df = result.statistic('sga', 'sample_size', 'normal_shifted', 'feature')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.array([[3067, 1966], [3041, 1926]]), decimal=5)
		# check variant_mean
		df = result.statistic('sga', 'variant_mean', 'normal_shifted', 'feature')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.array([[0.001221, 0.981820], [-0.012310, 0.989196]]), decimal=5)

	def test__trend__index_levels(self):
		"""
    Check if trend() returns the proper index levels
    """
		np.random.seed(0)
		metrics, metadata = generate_random_data()
		metrics['time_since_treatment'] = metrics['treatment_start_time']
		exp = Experiment('B', metrics, metadata, [4, 6])
		# Perform sga()
		result = exp.trend()
		# Check if all index levels are present
		index_levels = [
			pd.Index([u'normal_same', u'normal_shifted', u'normal_shifted_by_feature', u'normal_unequal_variance'],
					 dtype='object', name=u'metric'),
			pd.Index([u'-'], dtype='object', name=u'subgroup_metric'),
			pd.Index(range(10), dtype='object', name=u'time'),
			pd.Float64Index([], dtype='float64', name=u'subgroup'),
			pd.Index([u'sample_size', u'uplift', u'uplift_pctile', u'variant_mean'], dtype='object', name=u'statistic'),
			pd.Float64Index([2.5, 97.5], dtype='float64', name=u'pctile')
		]
		result_levels = list(result.df.index.levels)
		# Check if all index levels match expectation TODO: Make nice
		np.testing.assert_array_equal(index_levels[0], result_levels[0])
		np.testing.assert_array_equal(index_levels[1], result_levels[1])
		np.testing.assert_array_equal(index_levels[2], result_levels[2])
		np.testing.assert_array_equal(index_levels[3], result_levels[3])
		np.testing.assert_array_equal(index_levels[4], result_levels[4])
		np.testing.assert_array_equal(index_levels[5], result_levels[5])

	def test__trend__computation(self):
		"""
    Check if trend() functions properly
    """
		np.random.seed(0)
		metrics, metadata = generate_random_data()
		metrics['time_since_treatment'] = metrics['treatment_start_time']
		exp = Experiment('B', metrics, metadata, [4, 6])
		# Perform sga()
		result = exp.trend()

		# check uplift
		df = result.statistic('trend', 'uplift', 'normal_shifted')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-1.009421, -0.847400, -1.119885, -1.042597, -0.868819,
												 -1.091165, -0.952307, -1.028234, -0.978774, -0.985696]), decimal=5)
		# check pctile
		df = result.statistic('trend', 'uplift_pctile', 'normal_shifted')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-1.137482, -0.881360, -0.970678, -0.724122, -1.245795,
												 -0.993975, -1.178494, -0.906699, -0.993683, -0.743954, -1.225361,
												 -0.956969, -1.082180, -0.822435, -1.151715, -0.904753, -1.095209,
												 -0.862340, -1.109407, -0.861985]), decimal=5)
		# check samplesize
		df = result.statistic('trend', 'sample_size', 'normal_shifted')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.column_stack(([649, 595, 600, 590, 625, 602, 607, 608, 616, 616],
														[405, 401, 378, 362, 377, 369, 406, 392, 414, 388])), decimal=5)
		# check variant_mean
		df = result.statistic('trend', 'variant_mean', 'normal_shifted')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.column_stack(([0.005761, 0.057487, -0.067107, 0.001125, 0.093085,
														 -0.067894, -0.030500, -0.060996, 0.016257, -0.006091],
														[1.015182, 0.904887, 1.052778, 1.043721, 0.961904, 1.023271,
														 0.921807, 0.967238, 0.995031, 0.979605])), decimal=5)

	def test_trend_missing_time_resolved_data(self):
		"""Check if missing time-resolved data in trend analysis issues a warning"""
		result = self.data.trend()
		w = result.metadata['warnings']['Experiment.trend']
		self.assertTrue(isinstance(w, UserWarning))
		self.assertTrue(w.message == 'Empty data set entered to analysis.')

	def test_delta(self):
		"""
    Check if Experiment.delta() functions properly
    """
		# this should work
		self.assertTrue(isinstance(self.data, Experiment))  # check that the subclassing works

		self.assertTrue(self.data.baseline_variant == 'B')

		result = self.data.delta(kpi_subset=
								 [m for m in self.data.kpi_names if 'normal' in m])

		# check uplift
		df = result.statistic('delta', 'uplift', 'normal_same')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([0.033053]), decimal=5)
		# check pctile
		df = result.statistic('delta', 'uplift_pctile', 'normal_same')
		np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
									   np.array([-0.007135, 0.073240]), decimal=5)
		# check samplesize
		df = result.statistic('delta', 'sample_size', 'normal_same')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.array([[6108, 3892]]), decimal=5)
		# check variant_mean
		df = result.statistic('delta', 'variant_mean', 'normal_same')
		np.testing.assert_almost_equal(df.loc[:, 'value'],
									   np.array([[0.025219, -0.007833]]), decimal=5)

	def test_unequal_variance_warning_in_results(self):
		"""Check if the unequal variance warning message is persisted to the Results structure
    """
		result = self.data.delta(kpi_subset=['normal_unequal_variance'],
								 variant_subset=['A'])
		w = result.metadata['warnings']['Experiment.delta']
		self.assertTrue(isinstance(w, UserWarning))
		self.assertTrue(w.message == 'Sample variances differ too much to assume that population variances are equal.')


if __name__ == '__main__':
	unittest.main()
