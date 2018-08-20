import unittest
import warnings

import numpy as np
from scipy import stats

import expan.core.statistics as statx
from expan.core.util import find_value_by_key_with_condition
from .util import *

warnings.simplefilter('always')

class StatisticsTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data'
        self.samples = get_norm_temp_data(data_dir)
        self.metrics = get_framingham_data(data_dir)
        self.rand_s1 = np.random.normal(loc=0, size=1000)
        self.rand_s2 = np.random.normal(loc=0.1, size=1000)

    def tearDown(self):
        pass


class DeltaTestCases(StatisticsTestCase):
    def test__delta__not_providing_data_fails(self):
        """ Value error raised when not providing data. """
        # Check if error is raised for None data
        with self.assertRaises(ValueError):
            statx.delta(self.samples.temperature, None)
        # Check if error is raised for None data
        with self.assertRaises(ValueError):
            statx.delta(None, self.samples.temperature)

    def test__delta__computation_assumed_normal(self):
        """ Result of delta() assuming normality equals expected result. """
        # Computing delta assumed normal
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2],
            assume_normal=True)
        # Checking if mean has right value
        self.assertAlmostEqual(res.delta, -0.28923076923075541)

        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')

        # Checking if lower percentile has right value
        self.assertAlmostEqual(value025, -0.53963938185557114)
        # Checking if uper percentile has right value
        self.assertAlmostEqual(value975, -0.038822156605939739)
        # Checking if sample size 1 is correct
        self.assertEqual(res.treatment_statistics.sample_size, 65)
        # Checking if sample size 2 is correct
        self.assertEqual(res.control_statistics.sample_size, 65)

    def test__delta__2percentiles_no_tests(self):
        """ Percentiles of delta() for sga are corrected for no tests (1 as a default). """
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2])

        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')

        self.assertAlmostEqual(value025, -0.53963938185557114)
        self.assertAlmostEqual(value975, -0.038822156605939739)

    def test__delta__nan_handling(self):
        """ Test correct handling of nans. (ignored). """
        res = statx.delta(self.rand_s1, self.rand_s2)
        self.assertEqual(res.treatment_statistics.sample_size, 1000)
        self.assertEqual(res.control_statistics.sample_size, 1000)

        r1 = self.rand_s1.copy()
        r1[90:] = np.nan
        res = statx.delta(r1, self.rand_s2)

        self.assertAlmostEqual (res.delta, -0.1, 1)
        self.assertEqual       (res.treatment_statistics.sample_size, 90)
        self.assertEqual       (res.control_statistics.sample_size, 1000)

    def test__delta__computation_not_assumed_normal(self):
        """ Result of delta() not assuming normality equals expected result. """
        # Computing delta not assumed normal
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2],
            assume_normal=True)
        # Checking if mean has right value
        self.assertAlmostEqual(res.delta, -0.28923076923075541)

        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')

        # Checking if lower percentile has right value
        self.assertAlmostEqual(value025, -0.53963938185557114)
        # Checking if uper percentile has right value
        self.assertAlmostEqual(value975, -0.038822156605939739)
        # Checking if sample size 1 is correct
        self.assertEqual(res.treatment_statistics.sample_size, 65)
        # Checking if sample size 2 is correct
        self.assertEqual(res.control_statistics.sample_size, 65)


class SampleSizeTestCases(StatisticsTestCase):
    def test__sample_size__empty_list_numeric(self):
        """ Empty list returns 0. """
        self.assertEqual(statx.sample_size([]), 0)

    def test__sample_size__list_numeric(self):
        """ Result of sample_size() is number of elements of a list. """
        x = [1, 1, 2, 5, 8]
        self.assertEqual(statx.sample_size(x), 5)

    def test__sample_size__nparray_numeric_with_nan(self):
        """ Result of sample_size() is number of elements of numpy array minus number of NaNs. """
        x = np.array([1, 1, np.nan, 2, np.nan, 5, 8])
        self.assertEqual(statx.sample_size(x), 5)

    def test__sample_size__list_categorical(self):
        """ Result of sample_size() is number of elements of a list of categorical data. """
        x = ['1', '1', '3', '2', '6', '5', '8']
        self.assertEqual(statx.sample_size(x), 7)

    def test__sample_size__pdseries_categorical(self):
        """Result of sample_size() is number of elements of a pandas series with categorical data. """
        x = pd.Series(['1', '7', '2', '5', '8', '0'])
        self.assertEqual(statx.sample_size(x), 6)

    def test__sample_size__pdseries_categorical_with_na(self):
        """ Result of sample_size() is number of elements of a pandas series with categorical data including NAs. """
        x = ['1', '1', 'NA', '2', 'NA', '5', '8']
        self.assertEqual(statx.sample_size(x), 5)


class EstimateSampleSizeTestCases(StatisticsTestCase):
    def test__estimate_sample_size__dataframe(self):
        """ Result of estimate_sample_size() is a series of estimated sample sizes. """
        x = pd.DataFrame({'sample_1': [1, 7, 8, 9, 3, 4, 2, 0], 'sample_2': [5, 4, 3, 2, 2, 2, 1, 0]})
        res = statx.estimate_sample_size(x=x, mde=0.01, r=1.0)
        self.assertEqual(int(res['sample_1']), 197405)
        self.assertEqual(int(res['sample_2']), 142130)

    def test__estimate_sample_size__series(self):
        """ Result of estimate_sample_size() is estimated sample size. """
        x = pd.Series([1, 7, 8, 9, 3, 4, 2, 0])
        self.assertEqual(int(statx.estimate_sample_size(x=x, mde=0.01, r=1.0)), 197405)

    def test__estimate_sample_size__x_type_error(self):
        """ Method estimate_sample_size raises TypeError since x is a list. """
        x = [1, 7, 8, 9, 3, 4, 2, 0]
        self.assertRaises(TypeError, statx.estimate_sample_size, x=x, mde=0.01, r=1.0)

    def test__estimate_sample_size__r_value_error(self):
        """ Method estimate_sample_size raises ValueError since r is 0. """
        x = pd.Series([1, 7, 8, 9, 3, 4, 2, 0])
        self.assertRaises(ValueError, statx.estimate_sample_size, x=x, mde=0.01, r=0.0)

    def test__sample_size__all_nans(self):
        """ Result of sample_size() is number of elements of numpy array minus number of NaNs. """
        x = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        self.assertEqual(statx.sample_size(x), 0)


class BootstrapTestCases(StatisticsTestCase):
    def test__bootstrap__computation(self):
        """ Result of bootstrap() equals expected result. """
        # Computing bootstrap
        result1 = statx.bootstrap(self.samples.temperature, [0], min_observations=1)
        # Checking if lower percentile of result2 is correct
        self.assertAlmostEqual(result1[0][2.5], 98.1220, 2)
        # Checking if upper percentile of result2 is correct
        self.assertAlmostEqual(result1[0][97.5], 98.3708, 2)
        # Checking if no bootstrap data was passed
        self.assertIsNone(result1[1])

        # Defining data and computing bootstrap
        zero_3 = np.array([0., 0., 0.])
        one_3 = np.array([1., 1., 1.])
        result2 = statx.bootstrap(zero_3, one_3, min_observations=3)
        # Checking if lower percentile of result2 is correct
        self.assertEqual(result2[0][2.5], -1.0)
        # Checking if upper percentile of result2 is correct
        self.assertEqual(result2[0][97.5], -1.0)
        # Checking if no bootstrap data was passed
        self.assertIsNone(result2[1])

        # Defining data and computing bootstrap
        sample1 = self.samples.temperature[self.samples.gender == 1]
        sample2 = self.samples.temperature[self.samples.gender == 2]
        result3 = statx.bootstrap(sample1, sample2)
        # Checking if lower percentile of result3 is correct
        self.assertAlmostEqual(result3[0][2.5], -0.53384615384615619)
        # Checking if upper percentile of result3 is correct
        self.assertAlmostEqual(result3[0][97.5], -0.049192307692299965)
        # Checking if no bootstrap data was passed
        self.assertIsNone(result3[1])


class PooledStdTestCases(StatisticsTestCase):
    def test__pooled_std__variances_differ_too_much_error(self):
        """ Warning raised when variances differ too much. """
        with warnings.catch_warnings(record=True) as w:
            statx.pooled_std(0.25, 4, 0.5, 4)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue('variances differ' in str(w[-1].message))

    def test__pooled_std__computation(self):
        """ Result of pooled_std() equals expected result. """
        # Define subset of data for test
        sbp = self.metrics.loc['systolic_bp', :]
        # Computation of pooled standard deviation
        result1 = statx.pooled_std(sbp.men.s, sbp.men.n, sbp.women.s, sbp.women.n)
        # Checking if result1 is correct
        self.assertAlmostEqual(result1, 19, 1)

        # Computation of pooled standard deviation
        result2 = statx.pooled_std(9.7, 6, 12.0, 4)
        # Checking pooled_std() on small sample subset
        self.assertAlmostEqual(result2, 10.6, 1)


class NormalDifferenceTestCases(StatisticsTestCase):
    def test__normal_difference__computation(self):
        """ Result of normal_difference() equals expected result. """
        # Define subset of data for first test
        sbp = self.metrics.loc['systolic_bp', :]
        # computation of normal difference
        result1 = statx.normal_difference(sbp.men.m, sbp.men.s, sbp.men.n, sbp.women.m, sbp.women.s, sbp.women.n)
        # Checking if lower percentile of result1 is correct
        self.assertAlmostEqual(result1[2.5], 0.44582598543756413)
        # Checking if upper percentile of result1 is correct
        self.assertAlmostEqual(result1[97.5], 2.9541740145624127)

        # Define subset of data for second test
        clst = self.metrics.loc['serum_cholesterol', :]
        # Computation of normal difference
        result2 = statx.normal_difference(clst.men.m, clst.men.s, clst.men.n,
                                          clst.women.m, clst.women.s, clst.women.n)
        # Checking if lower percentile of result2 is correct
        self.assertAlmostEqual(result2[2.5], -17.159814380797162)
        # Checking if upper percentile of result2 is correct
        self.assertAlmostEqual(result2[97.5], -12.240185619202816)

        # test subsample of systolic blood pressure. Example from:
        # http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals5.html
        # Computation of normal difference
        result3 = statx.normal_difference(117.5, 9.7, 6, 126.8, 12., 4)
        # Checking if lower percentile of result3 is correct
        self.assertAlmostEqual(result3[2.5], -25.10960582643531)
        # Checking if upper percentile of result3 is correct
        self.assertAlmostEqual(result3[97.5], 6.5096058264353118)


class NormalSampleDifferenceTestCases(StatisticsTestCase):
    def test__normal_sample_difference__computation(self):
        # Defining data and computing normal difference
        sample1 = self.samples.temperature[self.samples.gender == 1]
        sample2 = self.samples.temperature[self.samples.gender == 2]
        # Computation of normal sample difference
        result1 = statx.normal_sample_difference(sample1, sample2)
        # Checking if lower percentile of result1 is correct
        self.assertAlmostEqual(result1[2.5], -0.53963938185557114)
        # Checking if upper percentile of result1 is correct
        self.assertAlmostEqual(result1[97.5], -0.038822156605939739)


class StatisticalPowerTestCases(StatisticsTestCase):
    def test_compute_statistical_power(self):
        float_precision = 2
        # Confirm with pre-computed value by hand via power analysis
        alpha = 0.05
        beta  = 0.2
        sigma = 1
        mean1 = 1
        mean2 = 0
        n1    = 13
        n2    = 12
        z_1_minus_alpha = stats.norm.ppf(1-alpha)

        power = statx.compute_statistical_power(mean1, sigma, n1, mean2, sigma, n2, z_1_minus_alpha)
        self.assertAlmostEqual(power, 1-beta, float_precision)

    def test_zero_pooled_std(self):
        power = statx.compute_statistical_power_from_samples([1, 1, 1, 1], [2, 2, 2, 2])
        self.assertEqual(power, -1)


class PValueTestCases(StatisticsTestCase):
    def test_compute_p_value(self):
        float_precision = 2
        # Confirm with pre-computed value by hand via looking up t table
        # Given the following values, pooled std and standard error should both be 1.
        # By looking up from two-tailed t-table of degree of freedom 2, t=4.303 will lead to p=0.05.
        sigma      = 1
        mean1      = 4.303
        mean2      = 0
        n1         = 2
        n2         = 2
        expected_p = 0.05

        p = statx.compute_p_value(mean1, sigma, n1, mean2, sigma, n2)
        self.assertAlmostEqual(p, expected_p, float_precision)


if __name__ == '__main__':
    unittest.main()
