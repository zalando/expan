import unittest
import warnings

import numpy as np

from expan.core.statistical_test import *
from expan.core.experiment import Experiment
from expan.core.util import generate_random_data, find_value_by_key_with_condition


class ExperimentTestCase(unittest.TestCase):
    """ Defines the setUp() and tearDown() functions for the statistics test cases."""

    def setUp(self):
        """ Load the needed data sets for all StatisticsTestCases and set the random
        seed so that randomized algorithms show deterministic behaviour."""
        np.random.seed(0)
        data, metadata = generate_random_data()
        self.data, self.metadata = data, metadata

        # simple statistical test
        self.kpi = KPI('normal_same')
        self.variants = Variants('variant', 'B', 'A')
        self.nonsense_variants = Variants('variant', 'C', 'D')
        self.test_normal_same = StatisticalTest(self.kpi, [], self.variants)
        self.test_nonsense_variant = StatisticalTest(self.kpi, [], self.nonsense_variants)

        # statistical test with derived kpi
        self.derived_kpi = DerivedKPI('derived_kpi_one', 'normal_same', 'normal_shifted')
        self.test_derived_kpi = StatisticalTest(self.derived_kpi, [], self.variants)


    def tearDown(self):
        """ Clean up after the test """
        pass

    def getExperiment(self):
        return Experiment(self.data, self.metadata)

    def assertNumericalEqual(self, a, b, decimals):
        self.assertEqual(round(a, decimals), round(b, decimals))


class ExperimentClassTestCases(ExperimentTestCase):

    def test_Experiment_constructor(self):
        experiment = self.getExperiment()
        self.assertTrue(isinstance(experiment.data, pd.DataFrame))
        self.assertTrue(isinstance(experiment.metadata, dict))

    def test_fixed_horizon_analyze_statistical_test(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same, testmethod='fixed_horizon')

        self.assertNumericalEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertNumericalEqual(lower_bound_ci, -0.007135, ndecimals)
        self.assertNumericalEqual(upper_bound_ci, 0.073240, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size,   3892)

        self.assertNumericalEqual(res.result.treatment_statistics.mean,  0.025219, ndecimals)
        self.assertNumericalEqual(res.result.control_statistics.mean,   -0.007833, ndecimals)

        self.assertNumericalEqual(res.result.statistical_power, 0.36401, ndecimals)
        self.assertEqual(res.test.kpi.name, 'normal_same')

    def test_group_sequential_delta(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same, testmethod='group_sequential')

        self.assertNumericalEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertNumericalEqual(lower_bound_ci, -0.007135, ndecimals)
        self.assertNumericalEqual(upper_bound_ci, 0.073240, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size, 3892)

        self.assertNumericalEqual(res.result.treatment_statistics.mean, 0.025219, ndecimals)
        self.assertNumericalEqual(res.result.control_statistics.mean, -0.007833, ndecimals)

        self.assertNumericalEqual(res.result.statistical_power, 0.36401, ndecimals)
        self.assertEqual(res.result.stop, False)
        self.assertEqual(res.test.kpi.name, 'normal_same')

    def test_bayes_factor_delta(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same,
                                                            testmethod='bayes_factor',
                                                            num_iters=2000)

        self.assertNumericalEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertNumericalEqual(lower_bound_ci, -0.00829, ndecimals)
        self.assertNumericalEqual(upper_bound_ci, 0.07127, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size, 3892)

        self.assertNumericalEqual(res.result.treatment_statistics.mean, 0.025219, ndecimals)
        self.assertNumericalEqual(res.result.control_statistics.mean, -0.007833, ndecimals)

        self.assertEqual(res.result.stop, True)
        self.assertEqual(res.test.kpi.name, 'normal_same')

    def test_bayes_precision_delta(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same,
                                                            testmethod='bayes_precision',
                                                            num_iters=2000)
        self.assertNumericalEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertNumericalEqual(lower_bound_ci, -0.00829, ndecimals)
        self.assertNumericalEqual(upper_bound_ci, 0.07127, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size, 3892)

        self.assertNumericalEqual(res.result.treatment_statistics.mean, 0.025219, ndecimals)
        self.assertNumericalEqual(res.result.control_statistics.mean, -0.007833, ndecimals)

        self.assertEqual(res.result.stop, True)

    def test_quantile_filtering_multiple_columns(self):
        exp = self.getExperiment()
        flags = exp._quantile_filtering(
            kpis=[
                'normal_same',
                'normal_shifted',
                'normal_shifted_by_feature',
                'normal_unequal_variance'
            ],
            percentile=99.0,
            threshold_type='upper'
        )
        self.assertEqual(len(flags[flags==True]), 386)

    def test_outlier_filtering_lower_threshold(self):
        exp = self.getExperiment()
        exp.outlier_filter(
            kpis=[
                'normal_same',
                'normal_shifted',
                'normal_shifted_by_feature',
                'normal_unequal_variance'
            ],
            percentile=0.1,
            threshold_type='lower'
        )
        self.assertEqual(len(self.data) - len(exp.data), exp.metadata['filtered_entities_number'])

    def test_outlier_filtering_unsupported_kpi(self):
        exp = self.getExperiment()
        with self.assertRaises(KeyError):
            exp.outlier_filter(kpis=['revenue'])

    def test_outlier_filtering_unsupported_percentile(self):
        exp = self.getExperiment()
        with self.assertRaises(ValueError):
            exp.outlier_filter(kpis=['normal_same'], percentile=101.0)

    def test_outlier_filtering_unsupported_threshold_kind(self):
        exp = self.getExperiment()
        with self.assertRaises(ValueError):
            exp.outlier_filter(kpis=['normal_same'], threshold_type='uppper')

    def test_outlier_filtering_high_filtering_percentage(self):
        exp = self.getExperiment()
        with warnings.catch_warnings(record=True) as w:
            exp.outlier_filter(kpis=['normal_same'], percentile=97.9)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

    # Test is_valid_for_analysis
    def test_is_valid_for_analysis(self):
        exp = self.getExperiment()
        is_valid = exp._is_valid_for_analysis(self.data, self.test_normal_same)
        self.assertTrue(is_valid)

        is_not_valid = exp._is_valid_for_analysis(self.data, self.test_nonsense_variant)
        self.assertFalse(is_not_valid)

    # Test _get_weights
    def test_get_weights(self):
        exp = self.getExperiment()
        res = exp._get_weights(self.data, self.test_normal_same, 'B')
        self.assertEqual(res, 1.0)

    # Test _get_weights for derived kpis
    def test_get_weights_derived_kpi(self):
        exp = self.getExperiment()
        self.derived_kpi.make_derived_kpi(self.data)
        res = exp._get_weights(self.data, self.test_derived_kpi, 'B')
        self.assertTrue(isinstance(res, pd.Series))

if __name__ == '__main__':
    unittest.main()
