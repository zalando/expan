import unittest
import numpy as np

from expan.core.results import *
from expan.core.statistical_test import *
from expan.core.util import generate_random_data


class StatisticalTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(41)
        self.data, self.metadata = generate_random_data()

        np.random.seed(41)
        self.control_statistics   = SampleStatistics(1000, 0.0, 1.0)
        self.treatment_statistics = SampleStatistics(1200, 1.0, 1.0)
        self.delta                = 1.0
        self.p                    = 0.04
        self.statistical_power    = 0.8
        self.confidence_interval  = {2.5: 0.1, 97.5: 1.1}
        self.stop                 = True

        self.corrected_p          = 0.02
        self.corrected_ci         = {1.0: -0.7, 99.0: 0.7}
        self.corrected_stop       = False

        self.simple_stats = SimpleTestStatistics(self.control_statistics,
                                                 self.treatment_statistics,
                                                 self.delta,
                                                 self.confidence_interval,
                                                 self.p,
                                                 self.statistical_power)

        self.simple_stats_corrected = SimpleTestStatistics(self.control_statistics,
                                                           self.treatment_statistics,
                                                           self.delta,
                                                           self.corrected_ci,
                                                           self.corrected_p,
                                                           self.statistical_power)

        self.es_stats = EarlyStoppingTestStatistics(self.control_statistics,
                                                    self.treatment_statistics,
                                                    self.delta,
                                                    self.confidence_interval,
                                                    self.p,
                                                    self.statistical_power,
                                                    self.stop)

        self.es_stats_corrected = EarlyStoppingTestStatistics(self.control_statistics,
                                                              self.treatment_statistics,
                                                              self.delta,
                                                              self.corrected_ci,
                                                              self.corrected_p,
                                                              self.statistical_power,
                                                              self.corrected_stop)

        variants = Variants('variant', 'control', 'treatment')
        self.correction_method = CorrectionMethod.BONFERRONI

        kpi = KPI('revenue')
        self.statistical_test = StatisticalTest(self.data, kpi, [], variants)
        self.statistical_test_result = StatisticalTestResult(self.statistical_test, self.simple_stats_corrected)

        test_result1 = StatisticalTestResult(self.statistical_test,
                                             CombinedTestStatistics(self.simple_stats, self.simple_stats_corrected))
        test_result2 = StatisticalTestResult(self.statistical_test,
                                             CombinedTestStatistics(self.es_stats, self.es_stats_corrected))
        test_results = [test_result1, test_result2]
        self.statistical_test_results = MultipleTestSuiteResult(test_results, self.correction_method)

    def tearDown(self):
        pass

    def test_simple_test_statistics(self):
        self.assertAlmostEqual(self.simple_stats.delta, 1.0)
        self.assertEqual(len(self.simple_stats.confidence_interval), 2)

    def test_early_stopping_test_statistics(self):
        self.assertAlmostEqual(self.es_stats.delta, 1.0)
        self.assertEqual(len(self.es_stats.confidence_interval), 2)
        self.assertTrue(self.es_stats.stop)

    def test_corrected_test_statistics_simple(self):
        corrected_test_statistics = CombinedTestStatistics(self.simple_stats, self.simple_stats_corrected)
        self.assertEqual(corrected_test_statistics.original_test_statistics.p, 0.04)
        self.assertEqual(corrected_test_statistics.corrected_test_statistics.p, 0.02)

    def test_corrected_test_statistics_early_stopping(self):
        corrected_test_statistics = CombinedTestStatistics(self.es_stats, self.es_stats_corrected)
        self.assertEqual(corrected_test_statistics.original_test_statistics.p, 0.04)
        self.assertEqual(corrected_test_statistics.corrected_test_statistics.p, 0.02)
        self.assertTrue(corrected_test_statistics.original_test_statistics.stop)
        self.assertFalse(corrected_test_statistics.corrected_test_statistics.stop)

    def test_corrected_test_statistics_type_mismatch(self):
        type1 = str(type(self.simple_stats))
        type2 = str(type(self.es_stats_corrected))
        error_msg = "Type mismatch for type " + type1 + " and " + type2
        with self.assertRaisesRegexp(RuntimeError, error_msg):
            CombinedTestStatistics(self.simple_stats, self.es_stats_corrected)

    def test_statistical_test_results(self):
        self.assertEqual(self.statistical_test_result.test.kpi.name, 'revenue')
        self.assertAlmostEqual(self.statistical_test_result.result.delta, 1.0)

    def test_multi_test_suite_results(self):
        self.assertEqual(len(self.statistical_test_results.results), 2)
        self.assertEqual(self.statistical_test_results.correction_method, self.correction_method)
