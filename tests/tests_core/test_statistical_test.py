import unittest
import numpy as np

from expan.core.statistical_test import *


class StatisticalTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(41)

    def tearDown(self):
        pass

    def test_setup_single_test(self):
        variants = Variants('variant', 'control', 'treatment')
        mobile = FeatureFilter('device_type', 'mobile')
        kpi = KPI('revenue')

        test_revenue_overall = StatisticalTest(kpi, [], variants)
        test_revenue_mobile = StatisticalTest(kpi, [mobile], variants)

        self.assertEqual(test_revenue_overall.kpi.name, "revenue")
        self.assertEqual(test_revenue_overall.variants.control_name, "control")
        self.assertEqual(test_revenue_overall.variants.treatment_name, "treatment")

        self.assertEqual(test_revenue_mobile.kpi.name, "revenue")
        self.assertEqual(test_revenue_mobile.variants.control_name, "control")
        self.assertEqual(test_revenue_mobile.variants.treatment_name, "treatment")
        self.assertEqual(test_revenue_mobile.features[0].column_name, "device_type")
        self.assertEqual(test_revenue_mobile.features[0].column_value, "mobile")

    def test_setup_multiple_test_suite(self):
        variants = Variants('variant', 'control', 'treatment')
        mobile = FeatureFilter('device_type', 'mobile')
        desktop = FeatureFilter('device_type', 'desktop')
        tablet = FeatureFilter('device_type', 'tablet')
        kpi = KPI('revenue')

        test_revenue_overall = StatisticalTest(kpi, [], variants)
        test_revenue_mobile = StatisticalTest(kpi, [mobile], variants)
        test_revenue_desktop = StatisticalTest(kpi, [desktop], variants)
        test_revenue_tablet = StatisticalTest(kpi, [tablet], variants)

        tests = [test_revenue_overall, test_revenue_mobile, test_revenue_desktop, test_revenue_tablet]
        multi_test_suite = StatisticalTestSuite(tests, MultipleTestingCorrectionMethod.benjamini_hochberg_correction)

        self.assertEqual(multi_test_suite.size, 4)
        self.assertEqual(multi_test_suite.correction_method,
                         MultipleTestingCorrectionMethod.benjamini_hochberg_correction)
