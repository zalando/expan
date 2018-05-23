import unittest
import numpy as np

from expan.core.statistical_test import *
from expan.core.util import generate_random_data


class StatisticalTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(41)
        self.data, self.metadata = generate_random_data()

        # simple statistical test
        self.test_kpi = KPI('normal_same')
        self.test_variants = Variants('variant', 'A', 'B')
        self.test_normal_same = StatisticalTest(self.data, self.test_kpi, [], self.test_variants)

    def tearDown(self):
        pass

    def test_setup_single_test(self):
        variants = Variants('variant', 'control', 'treatment')
        mobile = FeatureFilter('device_type', 'mobile')
        kpi = KPI('revenue')

        test_revenue_overall = StatisticalTest(self.data, kpi, [], variants)
        test_revenue_mobile = StatisticalTest(self.data, kpi, [mobile], variants)

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

        test_revenue_overall = StatisticalTest(self.data, kpi, [], variants)
        test_revenue_mobile = StatisticalTest(self.data, kpi, [mobile], variants)
        test_revenue_desktop = StatisticalTest(self.data, kpi, [desktop], variants)
        test_revenue_tablet = StatisticalTest(self.data, kpi, [tablet], variants)

        tests = [test_revenue_overall, test_revenue_mobile, test_revenue_desktop, test_revenue_tablet]
        multi_test_suite = StatisticalTestSuite(tests, CorrectionMethod.BH)

        self.assertEqual(multi_test_suite.size, 4)
        self.assertEqual(multi_test_suite.correction_method, CorrectionMethod.BH)

    def test_make_derived_kpi(self):
        numerator = "normal_same"
        denominator = "normal_shifted"
        derived_kpi_name = "derived_kpi_one"
        DerivedKPI(derived_kpi_name, numerator, denominator).make_derived_kpi(self.data)

        # checks if column with the derived kpi was created
        self.assertTrue(derived_kpi_name in self.data.columns)

        # checks if values of new columns are of type float
        self.assertTrue(all(isinstance(kpi_value, float) for kpi_value in self.data[derived_kpi_name]))

    def test_get_variant(self):
        control = self.test_normal_same.variants.get_variant(
            self.data, self.test_normal_same.variants.control_name)[self.test_normal_same.kpi.name]
        self.assertEqual(len(control), 6108)
        self.assertTrue(isinstance(control, pd.Series))
