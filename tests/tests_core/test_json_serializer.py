import unittest
import numpy as np

from expan.core.results import *
from expan.core.statistical_test import *
from tests.tests_core.util import get_test_data_revenue_order


class StatisticalTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(41)
        self.data = get_test_data_revenue_order()

    def tearDown(self):
        pass

    def test_serialize_single_result(self):
        kpi = KPI('revenue')
        variants = Variants('variant', 'control', 'treatment')
        control_statistics   = SampleStatistics(1000, 0.0, 1.0)
        treatment_statistics = SampleStatistics(1200, 1.0, 1.0)
        delta                = 1.0
        p                    = 0.04
        statistical_power    = 0.8
        confidence_interval  = {2.5: 0.1, 97.5: 1.1}

        statistical_test = StatisticalTest(self.data, kpi, [], variants)
        simple_stats = SimpleTestStatistics(control_statistics,
                                            treatment_statistics,
                                            delta,
                                            confidence_interval,
                                            p,
                                            statistical_power)
        statistical_test_result = StatisticalTestResult(statistical_test, simple_stats)

        js_result = statistical_test_result.toJson()  # no error/exception should be raise
        print(js_result)    # use pytest -s to check the output if needed

    def test_serialize_multi_test_suite_result(self):
        kpi = KPI('revenue')
        variants = Variants('variant', 'control', 'treatment')
        control_statistics   = SampleStatistics(1000, 0.0, 1.0)
        treatment_statistics = SampleStatistics(1200, 1.0, 1.0)
        delta                = 1.0
        p                    = 0.04
        statistical_power    = 0.8
        confidence_interval  = {2.5: 0.1, 97.5: 1.1}
        stop                 = True

        corrected_p          = 0.02
        corrected_ci         = {1.0: -0.7, 99.0: 0.7}
        corrected_stop       = False

        simple_stats = SimpleTestStatistics(control_statistics,
                                            treatment_statistics,
                                            delta,
                                            confidence_interval,
                                            p,
                                            statistical_power)

        simple_stats_corrected = SimpleTestStatistics(control_statistics,
                                                      treatment_statistics,
                                                      delta,
                                                      corrected_ci,
                                                      corrected_p,
                                                      statistical_power)

        es_stats = EarlyStoppingTestStatistics(control_statistics,
                                               treatment_statistics,
                                               delta,
                                               confidence_interval,
                                               p,
                                               statistical_power,
                                               stop)

        es_stats_corrected = EarlyStoppingTestStatistics(control_statistics,
                                                         treatment_statistics,
                                                         delta,
                                                         corrected_ci,
                                                         corrected_p,
                                                         statistical_power,
                                                         corrected_stop)

        mobile = FeatureFilter('device_type', 'mobile')
        desktop = FeatureFilter('device_type', 'desktop')
        statistical_test_mobile = StatisticalTest(self.data, kpi, [mobile], variants)
        statistical_test_desktop = StatisticalTest(self.data, kpi, [desktop], variants)

        test_result1 = StatisticalTestResult(statistical_test_mobile, CombinedTestStatistics(simple_stats, simple_stats_corrected))
        test_result2 = StatisticalTestResult(statistical_test_desktop, CombinedTestStatistics(es_stats, es_stats_corrected))
        test_results = [test_result1, test_result2]
        statistical_test_results = MultipleTestSuiteResult(test_results, CorrectionMethod.BH)

        js_result = statistical_test_results.toJson()  # no error/exception should be raise
        print(js_result)    # use pytest -s to check the output if needed
