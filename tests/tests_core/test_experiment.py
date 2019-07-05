from __future__ import division # so that, under Python 2.*, integer division results in real numbers

import unittest
import warnings

import numpy as np

from expan.core.results import CombinedTestStatistics
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

        feature_has = FeatureFilter('feature', 'has')
        feature_non = FeatureFilter('feature', 'non')
        feature_one = FeatureFilter('feature', 'feature that only has one data point')

        # simple statistical test
        self.kpi = KPI('normal_same')
        self.variants = Variants('variant', 'B', 'A')
        self.nonsense_variants = Variants('variant', 'C', 'D')
        self.test_normal_same = StatisticalTest(self.data, self.kpi, [], self.variants)
        self.test_nonsense_variant = StatisticalTest(self.data, self.kpi, [], self.nonsense_variants)
        self.test_normal_same_feature_non = StatisticalTest(self.data, self.kpi, [feature_non], self.variants)
        self.test_normal_same_feature_has = StatisticalTest(self.data, self.kpi, [feature_has], self.variants)
        self.test_normal_same_feature_one = StatisticalTest(self.data, self.kpi, [feature_one], self.variants)

        # statistical test with derived kpi
        self.derived_kpi = DerivedKPI('derived_kpi_one', 'normal_same', 'normal_shifted')
        self.test_derived_kpi = StatisticalTest(self.data, self.derived_kpi, [], self.variants)

        # small dummy data frame
        data_dummy = np.array([['index', 'entity', 'variant', 'normal_same', 'normal_shifted'],
                               [0, 1, 'A', 2.0, 1.0], [1, 2, 'B', 3.0, 2.0],
                               [2, 3, 'A', 3.0, 2.0], [3, 4, 'B', 2.5, 1.0]])
        self.data_dummy_df = pd.DataFrame(data=data_dummy[1:, 1:],
                                          columns=data_dummy[0, 1:]).convert_objects(convert_numeric=True)

        data_dummy_with_nan = np.array([['index', 'entity', 'variant', 'normal_same', 'normal_shifted'],
                               [0, 1, 'A', 2.0, 1.0], [1, 2, 'B', 3.0, 2.0],
                               [2, 3, 'A', 3.0, 2.0], [3, 4, 'B', 2.5, 1.0],
                               [4, 5, 'A', 0.0, 0.0], [5, 6, 'B', None, None]])
        self.data_dummy_df_with_nan = pd.DataFrame(data=data_dummy_with_nan[1:, 1:],
                                                   columns=data_dummy_with_nan[0, 1:]).convert_objects(convert_numeric=True)

        # statistical test suite
        self.suite_with_one_test = StatisticalTestSuite([self.test_normal_same])
        self.suite_with_one_test_correction = StatisticalTestSuite([self.test_normal_same], CorrectionMethod.BH)
        self.suite_with_two_tests = StatisticalTestSuite([self.test_normal_same, self.test_derived_kpi],
                                                         CorrectionMethod.BONFERRONI)
        self.suite_with_one_subgroup = StatisticalTestSuite([self.test_normal_same_feature_one])
        self.suite_with_three_subgroups = StatisticalTestSuite([self.test_normal_same_feature_non,
                                                                self.test_normal_same_feature_has,
                                                                self.test_normal_same_feature_one],
                                                               CorrectionMethod.BH)

        # small dummy data frames producing zero std and -1 statistical power
        data_dummy_zero_std = np.array([['index', 'entity', 'variant', 'normal_same', 'normal_shifted'],
                                        [0, 1, 'A', 1.0, 1.0], [1, 2, 'B', 2.0, 2.0],
                                        [2, 3, 'A', 1.0, 1.0], [3, 4, 'B', 2.0, 2.0]])
        self.data_dummy_zero_std = pd.DataFrame(data=data_dummy_zero_std[1:, 1:],
                                                columns=data_dummy_zero_std[0, 1:]).convert_objects(
            convert_numeric=True)
        self.test_normal_same_zero_std = StatisticalTest(self.data_dummy_zero_std, self.kpi, [], self.variants)
        self.suite_with_one_test_zero_std = StatisticalTestSuite([self.test_normal_same_zero_std])

        # small dummy data frames with all nan values
        self.data_dummy_all_nan = pd.DataFrame({
                        'entity' : [1,2,3,4],
                        'variant' : ['A','B','C','D'],
                        'normal_same' : [np.nan] * 4,
                        'normal_shifted' : [np.nan] * 4,
                            })
        self.test_normal_same_nan_data = StatisticalTest(self.data_dummy_all_nan, self.kpi, [], self.variants)
        self.suite_with_one_test_with_nan_data = StatisticalTestSuite([self.test_normal_same_nan_data])

    def tearDown(self):
        """ Clean up after the test """
        pass

    def getExperiment(self):
        return Experiment(self.metadata)


class ExperimentClassTestCases(ExperimentTestCase):

    def test_Experiment_constructor(self):
        experiment = self.getExperiment()
        self.assertTrue(isinstance(experiment.metadata, dict))


class StatisticalTestTestCases(ExperimentTestCase):
    """ Test the method analyze_statistical_test. """

    def test_fixed_horizon(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same, test_method='fixed_horizon')

        self.assertAlmostEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertAlmostEqual(lower_bound_ci, -0.007135, ndecimals)
        self.assertAlmostEqual(upper_bound_ci, 0.073240, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size,   3892)

        self.assertAlmostEqual(res.result.treatment_statistics.mean,  0.025219, ndecimals)
        self.assertAlmostEqual(res.result.control_statistics.mean,   -0.007833, ndecimals)

        self.assertAlmostEqual(res.result.statistical_power, 0.36401, ndecimals)
        self.assertEqual(res.test.kpi.name, 'normal_same')

    def test_group_sequential(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same, test_method='group_sequential')

        self.assertAlmostEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertAlmostEqual(lower_bound_ci, -0.007135, ndecimals)
        self.assertAlmostEqual(upper_bound_ci, 0.073240, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size, 3892)

        self.assertAlmostEqual(res.result.treatment_statistics.mean, 0.025219, ndecimals)
        self.assertAlmostEqual(res.result.control_statistics.mean, -0.007833, ndecimals)

        self.assertAlmostEqual(res.result.statistical_power, 0.36401, ndecimals)
        self.assertEqual(res.result.stop, False)
        self.assertEqual(res.test.kpi.name, 'normal_same')

    def test_bayes_factor_delta(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same,
                                                            test_method='bayes_factor',
                                                            num_iters=2000)

        self.assertAlmostEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertAlmostEqual(lower_bound_ci, -0.00829, ndecimals)
        self.assertAlmostEqual(upper_bound_ci, 0.07127, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size, 3892)

        self.assertAlmostEqual(res.result.treatment_statistics.mean, 0.025219, ndecimals)
        self.assertAlmostEqual(res.result.control_statistics.mean, -0.007833, ndecimals)

        self.assertEqual(res.result.stop, True)
        self.assertEqual(res.test.kpi.name, 'normal_same')

    def test_bayes_precision_delta(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test(self.test_normal_same,
                                                            test_method='bayes_precision',
                                                            num_iters=2000)
        self.assertAlmostEqual(res.result.delta, 0.033053, ndecimals)

        lower_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(res.result.confidence_interval, 'percentile', 97.5, 'value')
        self.assertAlmostEqual(lower_bound_ci, -0.00829, ndecimals)
        self.assertAlmostEqual(upper_bound_ci, 0.07127, ndecimals)

        self.assertEqual(res.result.treatment_statistics.sample_size, 6108)
        self.assertEqual(res.result.control_statistics.sample_size, 3892)

        self.assertAlmostEqual(res.result.treatment_statistics.mean, 0.025219, ndecimals)
        self.assertAlmostEqual(res.result.control_statistics.mean, -0.007833, ndecimals)

        self.assertEqual(res.result.stop, True)


class StatisticalTestSuiteTestCases(ExperimentTestCase):
    """ Test the method analyze_statistical_test_suite. """

    def test_one_test_in_suite(self):
        ndecimals = 5
        res = self.getExperiment().analyze_statistical_test_suite(self.suite_with_one_test)

        self.assertEqual(res.correction_method, CorrectionMethod.NONE)
        self.assertEqual(len(res.results), 1)
        simple_stat_res = res.results[0].result

        self.assertAlmostEqual(simple_stat_res.original_test_statistics.delta, 0.033053, ndecimals)
        lower_bound_ci = find_value_by_key_with_condition(simple_stat_res.original_test_statistics.confidence_interval,
                                                          'percentile', 2.5, 'value')
        upper_bound_ci = find_value_by_key_with_condition(simple_stat_res.original_test_statistics.confidence_interval,
                                                          'percentile', 97.5, 'value')
        self.assertAlmostEqual(lower_bound_ci, -0.007135, ndecimals)
        self.assertAlmostEqual(upper_bound_ci, 0.073240, ndecimals)

        self.assertEqual(simple_stat_res.original_test_statistics.treatment_statistics.sample_size, 6108)
        self.assertEqual(simple_stat_res.original_test_statistics.control_statistics.sample_size,   3892)

        self.assertAlmostEqual(simple_stat_res.original_test_statistics.treatment_statistics.mean,  0.025219, ndecimals)
        self.assertAlmostEqual(simple_stat_res.original_test_statistics.control_statistics.mean,   -0.007833, ndecimals)
        self.assertAlmostEqual(simple_stat_res.original_test_statistics.statistical_power, 0.36401, ndecimals)

        # corrected test statistics should be the same as original test statistics (checks some values)
        self.assertAlmostEqual(simple_stat_res.corrected_test_statistics.delta, 0.033053, ndecimals)
        self.assertAlmostEqual(simple_stat_res.corrected_test_statistics.statistical_power, 0.36401, ndecimals)
        self.assertEqual(simple_stat_res.corrected_test_statistics.treatment_statistics.sample_size, 6108)
        self.assertEqual(simple_stat_res.corrected_test_statistics.control_statistics.sample_size, 3892)

    def test_one_test_in_suite_with_wrong_correction(self):
        res = self.getExperiment().analyze_statistical_test_suite(self.suite_with_one_test_correction)
        self.assertEqual(res.correction_method, CorrectionMethod.NONE)
        self.assertEqual(len(res.results), 1)

    def test_two_tests_in_suite(self):
        ndecimals = 5

        res = self.getExperiment().analyze_statistical_test_suite(self.suite_with_two_tests)
        self.assertEqual(res.correction_method, CorrectionMethod.BONFERRONI)
        self.assertEqual(len(res.results), 2)

        res_normal_same = res.results[0]
        res_derived_kpi = res.results[1]

        self.assertEqual(res_normal_same.test.kpi.name, "normal_same")
        self.assertEqual(res_derived_kpi.test.kpi.name, "derived_kpi_one")
        self.assertTrue(isinstance(res_normal_same.result, CombinedTestStatistics))
        self.assertTrue(isinstance(res_derived_kpi.result, CombinedTestStatistics))

        power_normal_same_before = res_normal_same.result.original_test_statistics.statistical_power
        power_normal_same_corrected = res_normal_same.result.corrected_test_statistics.statistical_power
        self.assertAlmostEqual(power_normal_same_before, 0.36401, ndecimals)
        self.assertLess(power_normal_same_corrected, power_normal_same_before)

        power_derived_kpi_before = res_derived_kpi.result.original_test_statistics.statistical_power
        power_derived_kpi_corrected = res_derived_kpi.result.corrected_test_statistics.statistical_power
        self.assertAlmostEqual(power_derived_kpi_before, 0.3472, ndecimals)
        self.assertLess(power_derived_kpi_corrected, power_derived_kpi_before)

    def test_one_subgroup_in_suite(self):
        res = self.getExperiment().analyze_statistical_test_suite(self.suite_with_one_subgroup)
        self.assertEqual(res.correction_method, CorrectionMethod.NONE)
        self.assertEqual(len(res.results), 0)

    def test_three_subgroup_in_suite(self):
        """ Two subgroups contain valid data. One subgroup contains only one entity.
        When there is no enough data in one subgroup, which means one subgroup might contain zero
        or only one variant. (e.g. "browser" = "hack name"), it should ignore this subgroup and continue
        doing analysis with other subgroups.
        """
        res = self.getExperiment().analyze_statistical_test_suite(self.suite_with_three_subgroups)
        self.assertEqual(res.correction_method, CorrectionMethod.BH)
        self.assertEqual(len(res.results), 2)

        res_subgroup_feature_non = res.results[0]
        res_subgroup_feature_has = res.results[1]

        self.assertTrue(isinstance(res_subgroup_feature_non.result, CombinedTestStatistics))
        self.assertTrue(isinstance(res_subgroup_feature_has.result, CombinedTestStatistics))
        self.assertLess(res_subgroup_feature_non.result.corrected_test_statistics.statistical_power,
                        res_subgroup_feature_non.result.original_test_statistics.statistical_power)
        self.assertLess(res_subgroup_feature_has.result.corrected_test_statistics.statistical_power,
                        res_subgroup_feature_has.result.original_test_statistics.statistical_power)

    def test_analyze_statistical_test_suite_with_zero_std(self):
        res = self.getExperiment().analyze_statistical_test_suite(self.suite_with_one_test_zero_std)
        self.assertEqual(res.correction_method, CorrectionMethod.NONE)
        self.assertEqual(len(res.results), 0)

    def test_analyze_statistical_test_with_nan_data(self):
        res = self.getExperiment().analyze_statistical_test_suite(self.suite_with_one_test_with_nan_data)
        self.assertEqual(res.correction_method, CorrectionMethod.NONE)
        self.assertEqual(len(res.results), 0)


class OutlierFilteringTestCases(ExperimentTestCase):
    """ Test outlier filtering and quantile filtering. """

    kpi_names = [
        'normal_same',
        'normal_shifted',
        'normal_shifted_by_feature',
        'normal_unequal_variance'
    ]

    def test_quantile_filtering_multiple_columns(self):
        exp = self.getExperiment()
        flags = exp._quantile_filtering(
            self.data,
            kpis = self.kpi_names,
            thresholds = { kpi: ('upper', 99.0) for kpi in self.kpi_names }
        )
        self.assertEqual(len(flags[flags==True]), 386)
        filtered = [item[1] for item in list(zip(flags, self.data['variant'])) if item[0] == True]
        filtered_dict = dict((val, filtered.count(val)) for val in set(filtered))
        self.assertEqual(filtered_dict['A'], 88)
        self.assertEqual(filtered_dict['B'], 298)

    def test_outlier_filtering_automatic(self):
        exp = self.getExperiment()
        data = exp.outlier_filter(
            self.data,
            kpis = [KPI(kpi) for kpi in self.kpi_names],
        )
        self.assertEqual(len(self.data) - len(data), exp.metadata['filtered_entities_number'])
        self.assertEqual(exp.metadata['filtered_entities_number'], 389)

    def test_outlier_filtering_lower_threshold(self):
        exp = self.getExperiment()
        data = exp.outlier_filter(
            self.data,
            kpis = [KPI(kpi) for kpi in self.kpi_names],
            thresholds = { kpi: ('lower', 0.1) for kpi in self.kpi_names }
        )
        self.assertEqual(len(self.data) - len(data), exp.metadata['filtered_entities_number'])
        self.assertEqual(exp.metadata['filtered_entities_per_variant']['A'], 22)
        self.assertEqual(exp.metadata['filtered_entities_per_variant']['B'], 18)

    def test_outlier_filtering_unsupported_kpi(self):
        exp = self.getExperiment()
        with self.assertRaises(KeyError):
            exp.outlier_filter(self.data, kpis=[KPI('revenue')])

    def test_outlier_filtering_unsupported_percentile(self):
        exp = self.getExperiment()
        with self.assertRaises(ValueError):
            exp.outlier_filter(self.data, kpis=[KPI('normal_same')],
                               thresholds = {'normal_same': ('upper', 101.0)})

    def test_outlier_filtering_unsupported_threshold_kind(self):
        exp = self.getExperiment()
        with self.assertRaises(ValueError):
            exp.outlier_filter(self.data, kpis=[KPI('normal_same')],
                               thresholds = {'normal_same': ('upppper', 99.0)})

    def test_outlier_filtering_high_filtering_percentage(self):
        exp = self.getExperiment()
        with warnings.catch_warnings(record=True) as w:
            exp.outlier_filter(self.data, kpis=[KPI('normal_same')],
                               thresholds = {'normal_same': ('upper', 97.9)})
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

    def test_outlier_filtering_derived_kpi(self):
        my_kpis=[
            KPI('normal_same'),
            KPI('normal_shifted'),
            DerivedKPI('derived_kpi', 'normal_same', 'normal_shifted')
        ]
        exp = self.getExperiment()
        data = exp.outlier_filter(
            self.data,
            kpis=my_kpis,
            thresholds = {kpi.name: ('lower', 0.1) for kpi in my_kpis}
        )
        self.assertIn('derived_kpi', data.columns)

    def test_run_goodness_of_fit_test_true(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([230, 190, 190, 190, 230], ['A', 'B', 'C', 'D', 'E'])
        expected_freqs = pd.Series([206, 206, 206, 206, 206], ['A', 'B', 'C', 'D', 'E'])
        result = exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)
        self.assertEqual(result[0], True)
        self.assertAlmostEqual(result[1], 0.05357161207695437)

    def test_run_goodness_of_fit_test_false(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([230, 180, 170, 190, 230], ['A', 'B', 'C', 'D', 'E'])
        expected_freqs = pd.Series([250, 150, 100, 400, 100], ['A', 'B', 'C', 'D', 'E'])
        result = exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)
        self.assertEqual(result[0], False)
        self.assertAlmostEqual(result[1], 1.5123889771594655e-147)

    def test_run_goodness_of_fit_test_2_variants(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([170, 170], ['A', 'B'])
        expected_freqs = pd.Series([170, 170], ['A', 'B'])
        result = exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)
        self.assertEqual(result[0], True)
        self.assertAlmostEqual(result[1], 1.0)

    def test_run_goodness_of_fit_test_NaN_data(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([170, 170], ['A', np.nan])
        expected_freqs = pd.Series([170, 170], ['A', 'B'])
        with self.assertRaisesRegexp(ValueError, "Variant split check was cancelled since observed "
                                                 "or expected frequencies are less than 2."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_run_goodness_of_fit_test_1_variant_after_filter(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([180, 1, 2], ['A', 'B', 'C'])
        expected_freqs = pd.Series([170, 61, 61], ['A', 'B', 'C'])
        with self.assertRaisesRegexp(ValueError, "Variant split check was cancelled since observed "
                                                 "or expected frequencies are less than 2."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_test_run_goodness_of_fit_test_2_variants_after_filter(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([180, 184, 2], ['A', 'B', 'C'])
        expected_freqs = pd.Series([122, 122, 122], ['A', 'B', 'C'])
        result = exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)
        self.assertEqual(result[0], False)
        self.assertAlmostEqual(result[1], 1.5123889771594655e-14)

    def test_test_run_goodness_of_fit_test_one_valid_variant(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([170, 170], ['A', 'C'])
        expected_freqs = pd.Series([170, 170], ['A', 'B'])
        with self.assertRaisesRegexp(ValueError, "Variant split check was cancelled since observed "
                                                 "or expected frequencies are less than 2."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_test_run_goodness_of_fit_test_one_variant(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([180], ['A'])
        expected_freqs = pd.Series([170], ['A'])
        with self.assertRaisesRegexp(ValueError, "Variant split check was cancelled since observed "
                                                 "or expected frequencies are less than 2."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_test_run_goodness_of_fit_test_no_expected_freqs(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([180], ['A'])
        expected_freqs = pd.Series([], [])
        with self.assertRaisesRegexp(ValueError, "Variant split check was cancelled since expected "
                                                 "or observed frequencies are empty."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_test_run_goodness_of_fit_test_unequal_expected_observed_freqs(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series([180, 170], ['A', 'B'])
        expected_freqs = pd.Series([180], ['A'])
        with self.assertRaisesRegexp(ValueError, "Variant split check was cancelled since observed "
                                                 "or expected frequencies are less than 2."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_test_run_goodness_of_fit_test_None_freqs(self):
        exp = self.getExperiment()
        observed_freqs = None
        expected_freqs = None
        with self.assertRaisesRegexp(ValueError, "Observed and expected frequencies should be of type Series."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_test_run_goodness_of_fit_test_empty_input(self):
        exp = self.getExperiment()
        observed_freqs = pd.Series()
        expected_freqs = pd.Series()
        with self.assertRaisesRegexp(ValueError, "Variant split check was cancelled since expected "
                                                 "or observed frequencies are empty."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_test_run_goodness_of_fit_test_incorrect_input_1(self):
        exp = self.getExperiment()
        observed_freqs = {'A': 45, 'B': 35}
        expected_freqs = {'A': 45, 'B': 35}
        with self.assertRaisesRegexp(ValueError, "Observed and expected frequencies should be of type Series."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)

    def test_test_run_goodness_of_fit_test_incorrect_input_2(self):
        exp = self.getExperiment()
        observed_freqs = {'A': 45, 'B': 35}
        expected_freqs = ['A', 'B', 'C']
        with self.assertRaisesRegexp(ValueError, "Observed and expected frequencies should be of type Series."):
            exp.run_goodness_of_fit_test(observed_freqs, expected_freqs)


class HelperMethodsTestCases(ExperimentTestCase):
    """ Test other helper methods. """

    # Test is_valid_for_analysis
    def test_is_valid_for_analysis(self):
        exp = self.getExperiment()
        is_valid = exp._is_valid_for_analysis(self.data, self.test_normal_same)
        self.assertTrue(is_valid)

        is_not_valid = exp._is_valid_for_analysis(self.data, self.test_nonsense_variant)
        self.assertFalse(is_not_valid)


if __name__ == '__main__':
    unittest.main()
