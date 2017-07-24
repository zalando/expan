import unittest

import numpy as np
import pandas as pd

from expan.core.experiment import Experiment
# from expan.core.results import Results
from expan.core.util import generate_random_data, get_column_names_by_type
# from tests.tests_core.test_results import mock_results_object


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
        data, metadata = generate_random_data()
        self.column_names = list(set(data.columns) - set(['variant', 'entity']))
        self.numeric_column_names = get_column_names_by_type(data, np.number)

        self.data, self.metadata = data, metadata

    def tearDown(self):
        """
        Clean up after the test
        """
        # TODO: find out if we have to remove data manually
        pass


class ExperimentClassTestCases(ExperimentTestCase):
    """
    Test cases for the Experiment class functions.
    """

    # valid ones
    derived_kpi_1 = {'name'   : 'derived_kpi_1',
                     'formula': 'normal_same/normal_shifted'}

    derived_kpi_2 = {'name'   : 'derived_kpi_2',
                     'formula': 'normal_shifted/normal_same'}

    # bad ones
    derived_kpi_3 = {'name'   : 'derived_kpi_3',
                     'formula': 'normal_shifted/non_existing'}

    derived_kpi_4 = {'name'   : 'derived_kpi_4',
                     'formula': 'non_existing/normal_same'}


    def assertNumericalEqual(self, a, b, decimals):
        self.assertEqual(round(a, decimals), round(b, decimals))


    def getExperiment(self, report_kpi_names=None, derived_kpis=[]):
        return Experiment('B', self.data, self.metadata, report_kpi_names, derived_kpis)


    def test_constructor(self):
        self.getExperiment()

        with self.assertRaises(ValueError):
            experiment = self.getExperiment(self.column_names + ['non_existing'])

        self.getExperiment(self.column_names + [self.derived_kpi_1['name'],
                                                self.derived_kpi_2['name']],
                          [self.derived_kpi_1, self.derived_kpi_2])

        with self.assertRaises(ValueError):
            self.getExperiment(self.column_names + [self.derived_kpi_1['name'],
                                                    self.derived_kpi_3['name']],
                               [self.derived_kpi_1, self.derived_kpi_3])


        with self.assertRaises(ValueError):
            self.getExperiment(self.column_names + [self.derived_kpi_4['name'],
                                                    self.derived_kpi_2['name']],
                               [self.derived_kpi_4, self.derived_kpi_2])


##     def test_newDelta(self):
##         experiment = self.getExperiment(['normal_same'])
##         res = experiment.newDelta()
## 
##     def test__feature_check__computation(self):
##         """
##         Check if feature check is correctly performed on test data set
##         """
##         # Perform feature check
##         result = self.experiment.feature_check()
##         # check delta
##         df = result.statistic('fc', 'pre_treatment_diff', 'treatment_start_time')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-0.04587]), decimal=5)
##         # check pctile
##         df = result.statistic('fc', 'pre_treatment_diff_pctile', 'treatment_start_time')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-0.16244,  0.07069]), decimal=5)
##         # check samplesize
##         df = result.statistic('fc', 'sample_size', 'treatment_start_time')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[6108, 3892]]), decimal=5)
##         # check variant_mean
##         df = result.statistic('fc', 'variant_mean', 'treatment_start_time')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[4.50115,  4.54702]]), decimal=5)
## 
##         # check chi-square p-values
##         df = result.statistic('fc', 'chi_square_p', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[0.769093, 1]]), decimal=5)
## 
##         # check metadata is preserved
##         np.testing.assert_equal(True, all(item in result.metadata.items() for item in self.testmetadata.items()))
## 
##     def test__sga__index_levels(self):
##         """
##         Check if sga() returns the proper index levels
##         """
##         # Perform sga()
##         result = self.experiment.sga()
##         # Check if all index levels are present
##         index_levels = [
##             pd.Index([u'normal_same', u'normal_shifted', u'normal_shifted_by_feature', u'normal_unequal_variance'],
##                      dtype='object', name=u'metric'),
##             pd.Index([u'feature', u'treatment_start_time'], dtype='object', name=u'subgroup_metric'),
##             pd.Index([u'[0.0,2.0)', u'[2.0,4.0)', u'[4.0,7.0)', u'[7.0,9.0]', u'{has}', u'{non}'], dtype='object',
##                      name=u'subgroup'),
##             pd.Index([u'sample_size', u'uplift', u'uplift_pctile', u'variant_mean'], dtype='object', name=u'statistic'),
##             pd.Float64Index([2.5, 97.5], dtype='float64', name=u'pctile')
##         ]
##         result_levels = list(result.df.index.levels)
##         # Check if all index levels match expectation TODO: Make nice
##         np.testing.assert_array_equal(index_levels[0], result_levels[0])
##         np.testing.assert_array_equal(index_levels[1], result_levels[1])
##         np.testing.assert_array_equal(index_levels[2], result_levels[2])
##         np.testing.assert_array_equal(index_levels[3], result_levels[3])
##         np.testing.assert_array_equal(index_levels[4], result_levels[4])
## 
##     def test__sga__computation(self):
##         """
##         Check if sga() performs proper computation
##         """
##         # Perform sga()
##         result = self.experiment.sga()
##         # check uplift
##         df = result.statistic('sga', 'uplift', 'normal_shifted', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-0.980599, -1.001506]), decimal=5)
##         # check pctile
##         df = result.statistic('sga', 'uplift_pctile', 'normal_shifted', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-1.036771, -0.924426, -1.058535, -0.944478]), decimal=5)
##         # check samplesize
##         df = result.statistic('sga', 'sample_size', 'normal_shifted', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[3067, 1966], [3041, 1926]]), decimal=5)
##         # check variant_mean
##         df = result.statistic('sga', 'variant_mean', 'normal_shifted', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[0.001221, 0.981820], [-0.012310, 0.989196]]), decimal=5)
## 
##         # check metadata is preserved
##         np.testing.assert_equal(True, all(item in result.metadata.items() for item in self.testmetadata.items()))
## 
##     def test__trend__index_levels(self):
##         """
##         Check if trend() returns the proper index levels
##         """
##         np.random.seed(0)
##         metrics, metadata = generate_random_data()
##         metrics['time_since_treatment'] = metrics['treatment_start_time']
##         exp = Experiment('B', metrics, metadata, [4, 6])
##         # Perform sga()
##         result = exp.trend()
##         # Check if all index levels are present
##         index_levels = [
##             pd.Index([u'normal_same', u'normal_shifted', u'normal_shifted_by_feature', u'normal_unequal_variance'],
##                      dtype='object', name=u'metric'),
##             pd.Index([u'-'], dtype='object', name=u'subgroup_metric'),
##             pd.Index([str(x) for x in np.arange(10.)], dtype='object', name=u'time'),
##             pd.Float64Index([], dtype='float64', name=u'subgroup'),
##             pd.Index([u'sample_size', u'uplift', u'uplift_pctile', u'variant_mean'], dtype='object', name=u'statistic'),
##             pd.Float64Index([2.5, 97.5], dtype='float64', name=u'pctile')
##         ]
##         result_levels = list(result.df.index.levels)
##         # Check if all index levels match expectation TODO: Make nice
##         np.testing.assert_array_equal(index_levels[0], result_levels[0])
##         np.testing.assert_array_equal(index_levels[1], result_levels[1])
##         np.testing.assert_array_equal(index_levels[2], result_levels[2])
##         np.testing.assert_array_equal(index_levels[3], result_levels[3])
##         np.testing.assert_array_equal(index_levels[4], result_levels[4])
##         np.testing.assert_array_equal(index_levels[5], result_levels[5])
## 
##     def test__trend__computation(self):
##         """
##         Check if trend() functions properly
##         """
##         np.random.seed(0)
##         metrics, metadata = generate_random_data()
##         metrics['time_since_treatment'] = metrics['treatment_start_time']
##         exp = Experiment('B', metrics, metadata, [4, 6])
##         # Perform sga() with non-cumulative results
##         result = exp.trend(cumulative=False)
## 
##         # check uplift
##         df = result.statistic('trend', 'uplift', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-1.02175, -1.12275, -0.97611, -0.85916, -1.08909, -0.85667,
##                                                  -0.96769, -1.05598, -1.00399, -0.9566 ]), decimal=5)
##         # check pctile
##         df = result.statistic('trend', 'uplift_pctile', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-1.14696, -0.89654, -1.24874, -0.99677, -1.10311, -0.84911,
##                                                  -0.9903 , -0.72801, -1.21351, -0.96467, -0.98513, -0.7282 ,
##                                                  -1.09129, -0.84408, -1.18276, -0.9292 , -1.13086, -0.87713,
##                                                  -1.08261, -0.83059]), decimal=5)
##         # check samplesize
##         df = result.statistic('trend', 'sample_size', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.column_stack(([634, 637, 601, 565, 639, 595, 573, 584, 618, 662],
##                                                         [384, 362, 399, 404, 381, 370, 393, 400, 398, 401])), decimal=5)
##         # check variant_mean
##         df = result.statistic('trend', 'variant_mean', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.column_stack(([-0.052325, -0.024658, 0.033599, 0.008031, -0.075325,
##                                                          0.074325, -0.040660, 0.020174, 0.009113, 0.000389],
##                                                         [0.969424, 1.098095, 1.009712, 0.867187, 1.013762,
##                                                          0.930994, 0.927027, 1.076156, 1.013106, 0.956987])), decimal=5)
## 
##         # Perform sga() with cumulative results
##         result = exp.trend()
##         # check uplift
##         df = result.statistic('trend', 'uplift', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-1.021749, -1.070321, -1.039468, -0.993117, -1.012738,
##                                                  -0.987513, -0.984259, -0.993804, -0.994975, -0.990985]), decimal=5)
##         # check pctile
##         df = result.statistic('trend', 'uplift_pctile', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-1.146963, -0.896535, -1.159145, -0.981496, -1.112230,
##                                                  -0.966707, -1.056767, -0.929468, -1.069403, -0.956073,
##                                                  -1.039374, -0.935652, -1.032089, -0.936429, -1.038557,
##                                                  -0.949051, -1.037176, -0.952774, -1.031001, -0.950969]), decimal=5)
##         # check samplesize
##         df = result.statistic('trend', 'sample_size', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.column_stack(([634, 1271, 1872, 2437, 3076, 3671, 4244, 4828, 5446, 6108],
##                                                         [384, 746, 1145, 1549, 1930, 2300, 2693, 3093, 3491, 3892])),
##                                        decimal=5)
##         # check variant_mean
##         df = result.statistic('trend', 'variant_mean', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.column_stack(([-0.052325, -0.038459, -0.015325, -0.009910, -0.023500,
##                                                          -0.007644, -0.012102, -0.008198, -0.006233, -0.005515],
##                                                         [0.969424, 1.031862, 1.024143, 0.983207, 0.989239,
##                                                          0.979869, 0.972158, 0.985607, 0.988742, 0.985470])), decimal=5)
##         # check metadata is preserved
##         np.testing.assert_equal(True, all(item in result.metadata.items() for item in self.testmetadata.items()))
## 
##     def test_trend_missing_time_resolved_data(self):
##         """Check if missing time-resolved data in trend analysis issues a warning"""
##         result = self.experiment.trend()
##         w = result.metadata['warnings']['Experiment.trend']
##         self.assertTrue(isinstance(w, UserWarning))
##         self.assertTrue(w.args[0] == 'Empty data set entered to analysis.')
## 
    def test_fixed_horizon_delta(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='fixed_horizon')

        aStats = res['normal_same']['A']['delta_statistics']
        self.assertNumericalEqual(aStats['delta'],           0.033053, ndecimals)

        self.assertNumericalEqual(aStats['interval'][02.5], -0.007135, ndecimals)
        self.assertNumericalEqual(aStats['interval'][97.5],  0.073240, ndecimals)

        self.assertEqual(aStats['n_x'], 6108)
        self.assertEqual(aStats['n_y'], 3892)

        self.assertNumericalEqual(aStats['mu_x'],  0.025219, ndecimals)
        self.assertNumericalEqual(aStats['mu_y'], -0.007833, ndecimals)


    def test_fixed_horizon_delta_derived_kpis(self):
        self.getExperiment(self.numeric_column_names + [self.derived_kpi_1['name'],
                                                      self.derived_kpi_2['name']],
                           [self.derived_kpi_1, self.derived_kpi_2]).delta()


    def test_group_sequential_delta(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='group_sequential')

        aStats = res['normal_same']['A']['delta_statistics']
        self.assertNumericalEqual(aStats['delta'],           0.033053, ndecimals)

        self.assertNumericalEqual(aStats['interval'][02.5], -0.007135, ndecimals)
        self.assertNumericalEqual(aStats['interval'][97.5],  0.073240, ndecimals)

        self.assertEqual(aStats['n_x'], 6108)
        self.assertEqual(aStats['n_y'], 3892)

        self.assertNumericalEqual(aStats['mu_x'],  0.025219, ndecimals)
        self.assertNumericalEqual(aStats['mu_y'], -0.007833, ndecimals)


    def test_group_sequential_delta_derived_kpis(self):
        self.getExperiment(self.numeric_column_names + [self.derived_kpi_1['name'],
                                                      self.derived_kpi_2['name']],
                           [self.derived_kpi_1, self.derived_kpi_2]).delta('group_sequential')

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor_delta(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='bayes_factor', num_iters=2000)

        aStats = res['normal_same']['A']['delta_statistics']
        self.assertNumericalEqual(aStats['delta'], 0.033053, ndecimals)

        self.assertEqual(aStats['stop'],      True, ndecimals)
        self.assertEqual(aStats['num_iters'], 2000, ndecimals)

        #
        # this can result in different numerical values depending on Python version
        #
        # self.assertNumericalEqual(aStats['interval'][02.5], -0.007079081, ndecimals)
        # self.assertNumericalEqual(aStats['interval'][97.5],  0.072703576, ndecimals)

        self.assertEqual(aStats['n_x'], 6108)
        self.assertEqual(aStats['n_y'], 3892)

        self.assertNumericalEqual(aStats['mu_x'],  0.025219, ndecimals)
        self.assertNumericalEqual(aStats['mu_y'], -0.007833, ndecimals)


    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor_delta_derived_kpis(self):
        exp = self.getExperiment([self.derived_kpi_1['name']], [self.derived_kpi_1])
        res = exp.delta(method='bayes_factor', num_iters=2000)


    # @unittest.skip("sometimes takes too much time")
    def test_bayes_precision_delta(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='bayes_precision', num_iters=2000)

        aStats = res['normal_same']['A']['delta_statistics']
        self.assertNumericalEqual(aStats['delta'], 0.033053, ndecimals)

        self.assertEqual(aStats['stop'], True, ndecimals)
        self.assertEqual(aStats['num_iters'], 2000, ndecimals)

        #
        # this can result in different numerical values depending on Python version
        #
        # self.assertNumericalEqual(aStats['interval'][02.5], -0.007079081, ndecimals)
        # self.assertNumericalEqual(aStats['interval'][97.5],  0.072703576, ndecimals)

        self.assertEqual(aStats['n_x'], 6108)
        self.assertEqual(aStats['n_y'], 3892)

        self.assertNumericalEqual(aStats['mu_x'],  0.025219, ndecimals)
        self.assertNumericalEqual(aStats['mu_y'], -0.007833, ndecimals)


    # @unittest.skip("sometimes takes too much time")
    def test_bayes_precision_delta_derived_kpis(self):
        exp = self.getExperiment([self.derived_kpi_1['name']], [self.derived_kpi_1])
        res = exp.delta(method='bayes_precision', num_iters=2000)

        # self.getExperiment([self.derived_kpi_1['name']], [self.derived_kpi_1]).delta()


##     def test_delta_derived_kpis(self):
##         """
##         Check if Experiment.fixed_horizon_delta() functions properly for derived KPIs
##         """
##         # this should work
##         self.assertTrue(isinstance(self.experiment, Experiment))  # check that the subclassing works
##         self.assertTrue(self.experiment.baseline_variant == 'B')
## 
##         res = mock_results_object(self.experiment,
##                                   derived_kpis=[{'name': 'derived', 'formula': 'normal_same/normal_shifted'}])
##         result = self.experiment.delta('fixed_horizon', kpi_subset=['derived'],
##                                        derived_kpis=[{'name': 'derived', 'formula': 'normal_same/normal_shifted'}])
## 
##         # check uplift
##         df = result.statistic('delta', 'uplift', 'derived')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([0.308368]), decimal=5)
##         # check pctile
##         df = result.statistic('delta', 'uplift_pctile', 'derived')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-4.319602, 4.936339]), decimal=5)
##         # check samplesize
##         df = result.statistic('delta', 'sample_size', 'derived')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[6108, 3892]]), decimal=5)
##         # check variant_mean
##         df = result.statistic('delta', 'variant_mean', 'derived')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[0.376876, 0.068508]]), decimal=5)
## 
##         # check metadata is preserved
##         np.testing.assert_equal(True, all(item in result.metadata.items() for item in self.testmetadata.items()))
## 
##     def test_delta_derived_kpis_weighted(self):
##         """
##         Check if Experiment.fixed_horizon_delta() functions properly for derived KPIs using 
##         the weighted method.
##         """
##         # this should work
##         self.assertTrue(isinstance(self.experiment, Experiment))  # check that the subclassing works
##         self.assertTrue(self.experiment.baseline_variant == 'B')
## 
##         res = mock_results_object(self.experiment,
##                                   derived_kpis=[{'name': 'derived', 'formula': 'normal_same/normal_shifted'}],
##                                   weighted_kpis=['derived'])
##         result = self.experiment.delta('fixed_horizon', kpi_subset=['derived'],
##                                        derived_kpis=[{'name': 'derived', 'formula': 'normal_same/normal_shifted'}],
##                                        weighted_kpis=['derived'])
## 
##         # check uplift
##         df = result.statistic('delta', 'uplift', 'derived')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-4.564575]), decimal=5)
##         # check pctile
##         df = result.statistic('delta', 'uplift_pctile', 'derived')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-10.274232, 1.145082]), decimal=5)
##         # check samplesize
##         df = result.statistic('delta', 'sample_size', 'derived')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[6108, 3892]]), decimal=5)
##         # check variant_mean
##         df = result.statistic('delta', 'variant_mean', 'derived')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[-4.572524, -0.007949]]), decimal=5)
## 
##         # check metadata is preserved
##         np.testing.assert_equal(True, all(item in result.metadata.items() for item in self.testmetadata.items()))
## 
##     def test_unequal_variance_warning_in_results(self):
##         """
##         Check if the unequal variance warning message is persisted to the Results structure
##         """
##         res = mock_results_object(self.experiment, variant_subset=['A'])
##         result = self.experiment.fixed_horizon_delta(res, kpis_to_analyse=['normal_unequal_variance'])
##         w = result.metadata['warnings']['Experiment.delta']
##         self.assertTrue(isinstance(w, UserWarning))
##         self.assertTrue(w.args[0] == 'Sample variances differ too much to assume that population variances are equal.')
## 
##     def test__time_dependent_deltas__computation(self):
##         """
##         Check if time_dependent_deltas() functions properly
##         """
##         # Calculate result
##         result = self.experiment._time_dependent_deltas(self.experiment.metrics.reset_index()
##                                                         [['variant', 'time_since_treatment', 'normal_shifted']],
##                                                         variants=['A', 'B'])
##         # check uplift
##         df = result.statistic('trend', 'uplift', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-1.02175, -1.12275, -0.97611, -0.85916, -1.08909,
##                                                  -0.85667, -0.96769, -1.05598, -1.00399, -0.9566 ]), decimal=5)
##         # check pctile
##         df = result.statistic('trend', 'uplift_pctile', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-1.14696, -0.89654, -1.24874, -0.99677, -1.10311, -0.84911,
##                                                  -0.9903 , -0.72801, -1.21351, -0.96467, -0.98513, -0.7282 ,
##                                                  -1.09129, -0.84408, -1.18276, -0.9292 , -1.13086, -0.87713,
##                                                  -1.08261, -0.83059]), decimal=5)
##         # check samplesize
##         df = result.statistic('trend', 'sample_size', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.column_stack(([634, 637, 601, 565, 639, 595, 573, 584, 618, 662],
##                                                         [384, 362, 399, 404, 381, 370, 393, 400, 398, 401])), decimal=5)
##         # check variant_mean
##         df = result.statistic('trend', 'variant_mean', 'normal_shifted')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.column_stack(([-0.052325, -0.024658, 0.033599, 0.008031, -0.075325,
##                                                          0.074325, -0.040660, 0.020174, 0.009113, 0.000389],
##                                                         [0.969424, 1.098095, 1.009712, 0.867187, 1.013762,
##                                                          0.930994, 0.927027, 1.076156, 1.013106, 0.956987])), decimal=5)
## 
##     def test__subgroup_deltas__computation(self):
##         """
##         Check if subgroup_deltas() functions properly
##         """
##         # Calculate result
##         result = self.experiment._subgroup_deltas(self.experiment.metrics.reset_index()
##                                                   [['variant', 'feature', 'normal_shifted']], variants=['A', 'B'])
##         # check uplift
##         df = result.statistic('sga', 'uplift', 'normal_shifted', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-0.980599, -1.001506]), decimal=5)
##         # check pctile
##         df = result.statistic('sga', 'uplift_pctile', 'normal_shifted', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, ('value', 'A')],
##                                        np.array([-1.036771, -0.924426, -1.058535, -0.944478]), decimal=5)
##         # check samplesize
##         df = result.statistic('sga', 'sample_size', 'normal_shifted', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[3067, 1966], [3041, 1926]]), decimal=5)
##         # check variant_mean
##         df = result.statistic('sga', 'variant_mean', 'normal_shifted', 'feature')
##         np.testing.assert_almost_equal(df.loc[:, 'value'],
##                                        np.array([[0.001221, 0.981820], [-0.012310, 0.989196]]), decimal=5)
## 
## 
## if __name__ == '__main__':
##     unittest.main()
