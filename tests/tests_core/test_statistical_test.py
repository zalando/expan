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

        test_revenue_overall = StatisticalTest('revenue', [], variants)
        test_revenue_mobile = StatisticalTest('revenue', [mobile], variants)

        self.assertEqual(test_revenue_overall.kpi_name, "revenue")
        self.assertEqual(test_revenue_overall.variants.control_name, "control")
        self.assertEqual(test_revenue_overall.variants.treatment_name, "treatment")

        self.assertEqual(test_revenue_mobile.kpi_name, "revenue")
        self.assertEqual(test_revenue_mobile.variants.control_name, "control")
        self.assertEqual(test_revenue_mobile.variants.treatment_name, "treatment")
        self.assertEqual(test_revenue_mobile.features[0].column_name, "device_type")
        self.assertEqual(test_revenue_mobile.features[0].column_value, "mobile")

    def test_setup_multiple_test_suite(self):
        variants = Variants('variant', 'control', 'treatment')
        mobile = FeatureFilter('device_type', 'mobile')
        desktop = FeatureFilter('device_type', 'desktop')
        tablet = FeatureFilter('device_type', 'tablet')

        test_revenue_overall = StatisticalTest('revenue', [], variants)
        test_revenue_mobile = StatisticalTest('revenue', [mobile], variants)
        test_revenue_desktop = StatisticalTest('revenue', [desktop], variants)
        test_revenue_tablet = StatisticalTest('revenue', [tablet], variants)

        tests = [test_revenue_overall, test_revenue_mobile, test_revenue_desktop, test_revenue_tablet]
        multi_test_suite = MultipleTestSuite(tests, MultipleTestingCorrectionMethod.benjamini_hochberg_correction)

        self.assertEqual(multi_test_suite.size, 4)
        self.assertEqual(multi_test_suite.correction_method,
                         MultipleTestingCorrectionMethod.benjamini_hochberg_correction)
