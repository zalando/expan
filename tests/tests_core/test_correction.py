import unittest

import numpy as np
from expan.core.results import *
from expan.core.correction import *
from expan.core.statistical_test import *
from expan.core.util import find_value_by_key_with_condition


class CorrectionTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(41)
        self.control_statistics = SampleStatistics(1000, 0.0, 1.0)
        self.treatment_statistics = SampleStatistics(1200, 1.0, 1.0)
        self.delta = 1.0
        self.p1 = 0.04
        self.p2 = 0.05
        self.statistical_power = 0.8
        self.confidence_interval = {2.5: 0.1, 97.5: 1.1}
        self.stop = True

        self.simple_stats = SimpleTestStatistics(self.control_statistics,
                                                 self.treatment_statistics,
                                                 self.delta,
                                                 self.confidence_interval,
                                                 self.p1,
                                                 self.statistical_power)

        self.es_stats = EarlyStoppingTestStatistics(self.control_statistics,
                                                    self.treatment_statistics,
                                                    self.delta,
                                                    self.confidence_interval,
                                                    self.p2,
                                                    self.statistical_power,
                                                    self.stop)

        variants = Variants('variant', 'control', 'treatment')
        self.correction_method = "bf"

        kpi_revenue = KPI('revenue')
        self.statistical_test = StatisticalTest(kpi_revenue, [], variants)
        self.statistical_test_result_1 = StatisticalTestResult(self.statistical_test, self.simple_stats)

        kpi_ctr = KPI('CTR')
        self.statistical_test = StatisticalTest(kpi_ctr, [], variants)
        self.statistical_test_result_2 = StatisticalTestResult(self.statistical_test, self.es_stats)

        test_results = [self.statistical_test_result_1, self.statistical_test_result_2]
        self.statistical_test_results = MultipleTestSuiteResult(test_results, self.correction_method)

    def tearDown(self):
        pass

    def test_add_corrected_test_statistics(self):
        corrected_test_results = add_corrected_test_statistics(self.statistical_test_results)
        corrected_stats_1 = corrected_test_results.statistical_test_results[0].result.corrected_test_statistics
        corrected_stats_2 = corrected_test_results.statistical_test_results[1].result.corrected_test_statistics

        self.assertEqual(corrected_stats_1.p, 0.08)
        self.assertEqual(corrected_stats_2.p, 0.1)
