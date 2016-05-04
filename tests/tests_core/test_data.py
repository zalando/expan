import unittest
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from expan.core.experimentdata import ExperimentData


def generate_random_data():
	np.random.seed(42)
	size = 10000

	test_data_frame = pd.DataFrame()
	test_data_frame['entity'] = range(size)
	test_data_frame['variant'] = np.random.choice(['A', 'B'], size=size, p=[0.6, 0.4])

	test_data_frame['normal_same'] = np.random.normal(size=size)
	test_data_frame['normal_shifted'] = np.random.normal(size=size)

	test_data_frame.loc[test_data_frame['variant'] == 'B', 'normal_shifted'] \
		= np.random.normal(loc=1.0, size=test_data_frame['normal_shifted'][test_data_frame['variant'] == 'B'].shape[0])

	test_data_frame['feature'] = np.random.choice(['has', 'non'], size=size)

	test_data_frame['normal_shifted_by_feature'] = np.random.normal(size=size)

	randdata = np.random.normal(loc=1.0, size=size)
	ii = (test_data_frame['variant'] == 'B') & (test_data_frame['feature'] == 'has')

	with warnings.catch_warnings(record=True) as w:
		# ignore the 'flat[index' warning that comes out of pandas (and is
		# not ours to fix)
		warnings.simplefilter('ignore', DeprecationWarning)

		test_data_frame.loc[ii, 'normal_shifted_by_feature'] = randdata

	test_data_frame['treatment_start_time'] = np.random.choice(range(10), size=size)

	test_data_frame['normal_unequal_variance'] = np.random.normal(size=size)
	test_data_frame.loc[test_data_frame['variant'] == 'B', 'normal_unequal_variance'] \
		= np.random.normal(scale=10,
						   size=test_data_frame['normal_unequal_variance'][test_data_frame['variant'] == 'B'].shape[0])

	metadata = {
		'primary_KPI': 'normal_shifted',
		'source': 'simulated',
		'experiment': 'random_data_generation'
	}

	return test_data_frame, metadata


class DataTestCase(unittest.TestCase):
	def setUp(self):
		"""
    Load the needed datasets for all StatisticsTestCases and set the random
    seed so that randomized algorithms show deterministic behaviour.
    """
		# np.random.seed(0)
		self.metrics, self.metadata = generate_random_data()

	def tearDown(self):
		"""
    Clean up after the test
    """
		# TODO: find out if we have to remove data manually
		pass

	def test_create_with_insufficient_data(self):
		# should not work:
		with self.assertRaises(KeyError):
			ExperimentData(
				pd.DataFrame(columns=['entity', 'variant']),
				metadata={'experiment': 'test', 'source': 'none'}
			)
		with self.assertRaises(KeyError):
			ExperimentData(
				pd.DataFrame(columns=['entity', 'variant', 'plums']),
				metadata={'experiment': 'test', 'source': 'none', 'primary_KPI': 'plums'}
			)
		with self.assertRaises(ValueError):
			ExperimentData(None)
		# with self.assertRaises(KeyError):
		#	ExperimentData(pd.DataFrame())
		with self.assertRaises(KeyError):
			ExperimentData(
				pd.DataFrame(columns=['entity', 'variant']),
				metadata=None)
		# with self.assertRaises(KeyError):
		# 	ExperimentData(
		# 		pd.DataFrame(columns=['entity', 'treatment_start']),
		# 		metadata={'experiment': 'fesf', 'source': 'random'},
		# 	)
		with self.assertRaises(KeyError):
			ExperimentData(
				pd.DataFrame(columns=['variant', 'treatment_start']),
				metadata={'experiment': 'fesf', 'source': 'random'}
			)
		with self.assertRaises(KeyError):
			ExperimentData(
				pd.DataFrame(columns=['variant', 'entity', 'treatment_start']),
				metadata={
					'experiment': 'fesf',
					'source': 'random',
					'primary_KPI': 'something_not_there'}
			)

	def test_data_generation(self):
		df, md = generate_random_data()
		A = ExperimentData(df, md)
		self.assertIsNotNone(A.kpis)
		self.assertIsNotNone(A.features)
		self.assertIsNotNone(A.metadata)

		# also test incomplete info again
		with self.assertRaises(KeyError):
			ExperimentData(df.drop('entity', axis=1), md)
		# with self.assertRaises(KeyError):
		# 	ExperimentData(df.drop('variant', axis=1), md)

	def test_direct_indexing(self):
		A = ExperimentData(*generate_random_data())
		# this should work
		normal_shifted = A[['normal_shifted', 'feature']]

		# this should not
		with self.assertRaises(KeyError):
			normal_shifted = A[['normal_shifted', 'non_existent_feature']]

	def test_initialize_without_kpi_with_feature(self):
		"""Initialize ExperimentData with metrics=None, features=DF"""
		metrics, _ = generate_random_data()
		meta = {
			'source': 'simulated',
			'experiment': 'random_data_generation'
		}
		D = ExperimentData(None, meta, metrics)

	def test_initialize_without_kpi_without_feature(self):
		"""Initialize ExperimentData with metrics=None, features=[]/'default'"""
		meta = {
			'source': 'simulated',
			'experiment': 'random_data_generation'
		}

		with self.assertRaises(ValueError):
			D = ExperimentData(None, meta, [])
		with self.assertRaises(ValueError):
			D = ExperimentData(None, meta, 'default')

	def test_initialize_with_metric_with_feature_default(self):
		"""Initialize ExperimentData with metrics=DF, features='default'"""
		# metrics,meta = generate_random_data()
		D = ExperimentData(self.metrics, self.metadata, 'default')

	def test_initialize_with_metric_with_feature_list(self):
		"""Initialize ExperimentData with metrics=DF, features=list"""
		metrics, meta = generate_random_data()
		D = ExperimentData(metrics, meta, [])
		D = ExperimentData(metrics, meta, [4, 6])

	def test_initialize_with_metric_with_feature_df(self):
		"""Initialize ExperimentData with metrics=DF, features=DF"""
		metrics, meta = generate_random_data()
		features = deepcopy(metrics)
		D = ExperimentData(metrics, meta, features)

	def test_init_with_aggregated_kpi(self):
		"""Initialize ExperimentData with aggregated KPI data"""
		D = ExperimentData(self.metrics, self.metadata, [4])
		self.assertIsNone(D.kpis_time)

	def test_init_with_time_resolved_kpi(self):
		"""Initialize ExperimentData with time-resolved KPI data"""
		n = 5  # replicate raw data and create synthetic time domain
		metrics_time = self.metrics.loc[np.repeat(self.metrics.index.values, n)]
		metrics_time.loc[:, 'time_since_treatment'] = np.tile(range(5), self.metrics.shape[0])

		D = ExperimentData(metrics_time, self.metadata, [4])
		self.assertIsNotNone(D.kpis_time)
		self.assertEquals(D.kpis.shape[0] * n, D.kpis_time.shape[0])


if __name__ == '__main__':
	unittest.main()
