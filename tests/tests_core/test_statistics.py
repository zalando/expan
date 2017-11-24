import unittest
import warnings

import numpy as np
from scipy import stats

import expan.core.statistics as statx
from expan.core.util import find_list_of_dicts_element
from .util import *


class StatisticsTestCase(unittest.TestCase):
    """
      Defines the setUp() and tearDown() functions for the statistics test cases.
      """

    def setUp(self):
        """
        Load the needed datasets for all StatisticsTestCases and set the random
        seed so that randomized algorithms show deterministic behaviour.
        """
        np.random.seed(0)
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data'

        self.samples = get_norm_temp_data(data_dir)
        self.metrics = get_framingham_data(data_dir)

        self.rand_s1 = np.random.normal(loc=0, size=1000)
        self.rand_s2 = np.random.normal(loc=0.1, size=1000)

    def tearDown(self):
        """
        Clean up after the test
        """
        pass


class DeltaTestCases(StatisticsTestCase):
    """
      Test cases for the delta() function in core.statistics.
      """

    def test__delta__not_providing_data_fails(self):
        """
        Value error raised when not providing data.
        """
        # Check if error is raised for None data
        with self.assertRaises(ValueError):
            statx.delta(self.samples.temperature, None)
        # Check if error is raised for None data
        with self.assertRaises(ValueError):
            statx.delta(None, self.samples.temperature)

    def test__delta__computation_assumed_normal(self):
        """
        Result of delta() assuming normality equals expected result.
        """
        # Computing delta assumed normal
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2],
            percentiles=[2.5, 97.5],
            assume_normal=True)
        # Checking if mean has right value
        self.assertAlmostEqual(res['delta'], -0.28923076923075541)

        value025 = find_list_of_dicts_element(res['confidence_interval'], 'percentile',  2.5, 'value')
        value975 = find_list_of_dicts_element(res['confidence_interval'], 'percentile', 97.5, 'value')

        # Checking if lower percentile has right value
        self.assertAlmostEqual(value025, -0.53770569567692295)
        # Checking if uper percentile has right value
        self.assertAlmostEqual(value975, -0.040755842784587965)
        # Checking if sample size 1 is correct
        self.assertEqual(res['treatment_sample_size'], 65)
        # Checking if sample size 2 is correct
        self.assertEqual(res['control_sample_size'], 65)

    def test__delta__2percentiles_25_tests(self):
        """
        Percentiles of delta() for sga are corrected for 25 tests
        """
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2],
            percentiles=[2.5, 97.5],
            assume_normal=True, num_tests=25)

        value025 = find_list_of_dicts_element(res['confidence_interval'], 'percentile', 0.1, 'value')
        value975 = find_list_of_dicts_element(res['confidence_interval'], 'percentile', 99.9, 'value')

        self.assertAlmostEqual(value025, -0.68544085117601006)
        self.assertAlmostEqual(value975, 0.1069793127145035)

    def test__delta__2percentiles_no_tests(self):
        """
        Percentiles of delta() for sga are corrected for no tests (1 as a default)
        """
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2],
            percentiles=[2.5, 97.5],
            assume_normal=True, num_tests=1)

        value025 = find_list_of_dicts_element(res['confidence_interval'], 'percentile', 2.5, 'value')
        value975 = find_list_of_dicts_element(res['confidence_interval'], 'percentile', 97.5, 'value')

        self.assertAlmostEqual(value025, -0.53770569567692295)
        self.assertAlmostEqual(value975, -0.040755842784587965)

    def test__delta__nan_handling(self):
        """
        Test correct handling of nans. (ignored)
        """
        res = statx.delta(self.rand_s1, self.rand_s2)
        self.assertEqual(res['treatment_sample_size'], 1000)
        self.assertEqual(res['control_sample_size'], 1000)

        r1 = self.rand_s1.copy()
        r1[90:] = np.nan
        res = statx.delta(r1, self.rand_s2)

        self.assertAlmostEqual (res['delta'], -0.1, 1)
        self.assertEqual       (res['treatment_sample_size'],    90)
        self.assertEqual       (res['control_sample_size'],    1000)

    def test__delta__computation_not_assumed_normal(self):
        """
        Result of delta() not assuming normality equals expected result.
        """
        # Computing delta not assumed normal
        res = statx.delta(
            self.samples.temperature[self.samples.gender == 1],
            self.samples.temperature[self.samples.gender == 2],
            percentiles=[2.5, 97.5],
            assume_normal=True)
        # Checking if mean has right value
        self.assertAlmostEqual(res['delta'],          -0.28923076923075541)

        value025 = find_list_of_dicts_element(res['confidence_interval'], 'percentile',  2.5, 'value')
        value975 = find_list_of_dicts_element(res['confidence_interval'], 'percentile', 97.5, 'value')

        # Checking if lower percentile has right value
        self.assertAlmostEqual(value025, -0.53770569567692295)
        # Checking if uper percentile has right value
        self.assertAlmostEqual(value975, -0.040755842784587965)
        # Checking if sample size 1 is correct
        self.assertEqual(res['treatment_sample_size'], 65)
        # Checking if sample size 2 is correct
        self.assertEqual(res['control_sample_size'], 65)


class ChiSquareTestCases(StatisticsTestCase):
    """
      Test cases for the chi_square() function in core.statistics.
      """

    def test__chi_square__not_providing_data_fails(self):
        """
        Value error raised when not providing data.
        """
        # Check if error is raised for None data
        with self.assertRaises(ValueError):
            statx.chi_square(self.samples.temperature, None)
        # Check if error is raised for None data
        with self.assertRaises(ValueError):
            statx.chi_square(None, self.samples.temperature)

    def test__chi_square__computation_same_data(self):
        """
        Check if p-value is 1.0 for same data entered twice.
        """
        # Computation of chi-square p-value
        self.assertEqual(1.0,
                         statx.chi_square(self.samples.temperature,
                                          self.samples.temperature)[0])

    def test__chi_square__computation_different_data(self):
        """
        Check if p-value is correct for test data from pandas manual page.
        """
        # Create test data:
        a = ['A'] * 16 + ['B'] * 18 + ['C'] * 16 + ['D'] * 14 + ['E'] * 12 + ['F'] * 12
        b = ['A'] * 16 + ['B'] * 16 + ['C'] * 16 + ['D'] * 16 + ['E'] * 16 + ['F'] * 8
        # Computation of chi-square p-value
        self.assertAlmostEqual(0.89852623940266074, statx.chi_square(a, b)[0])

    def test__chi_square__computation_different_data_as_in_statistics_book(self):
        """
        Check if p-value is correct for test data from statistics book
        Fahrmeir et al. (2007) pp. 463.
        """
        # Create test data:
        a = ['nein'] * 139 + ['gut'] * 348 + ['mittel'] * 213
        b = ['nein'] * 135 + ['gut'] * 46 + ['mittel'] * 119
        # Computation of chi-square p-value
        p, chisq, nattr = statx.chi_square(a, b)
        self.assertAlmostEqual(116.851, chisq, delta=0.001)
        self.assertAlmostEqual(0.0, p, delta=0.00000000001)

    def test__chi_square__computation_different_data_as_in_open_statistics_book(self):
        """
        Check if p-value is correct for test data from
        open statistics book 3rd ed pp. 299.
        (https://www.openintro.org/stat/textbook.php)
        """
        # Create test data:
        a = ['cu'] * 3511 + ['t1'] * 1749 + ['t2'] * 1818
        b = ['cu'] * 1489 + ['t1'] * 751 + ['t2'] * 682
        # Computation of chi-square p-value
        p, chisq, nattr = statx.chi_square(a, b)
        self.assertAlmostEqual(6.120, chisq, delta=0.001)
        self.assertAlmostEqual(0.0469, p, delta=0.0001)

    def test__chi_square__computation_one_bin_not_present(self):
        """
        Check if p-value is correct for test data from pandas manual page.
        """
        # Create test data:
        a = ['A'] * 16 + ['B'] * 18 + ['C'] * 16 + ['D'] * 14 + ['E'] * 12 + ['F'] * 12
        b = ['A'] * 16 + ['B'] * 16 + ['C'] * 16 + ['D'] * 16 + ['E'] * 16 + ['F'] * 8
        # Computation of chi-square p-value (a is shortened)
        self.assertAlmostEqual(0.94879980715092971, statx.chi_square(a[0:-12], b)[0])
        # Computation of chi-square p-value (b is shortened)
        self.assertAlmostEqual(0.94879980715092971, statx.chi_square(a, b[0:-8])[0])

    def test__chi_square__computation_symmetric(self):
        """
        Check if p-value is roughly symmetric.
        """
        # Create test data:
        a = ['A'] * 16 + ['B'] * 18 + ['C'] * 16 + ['D'] * 14 + ['E'] * 12 + ['F'] * 12
        b = ['A'] * 16 + ['B'] * 16 + ['C'] * 16 + ['D'] * 16 + ['E'] * 16 + ['F'] * 8
        # Computation of chi-square p-value (a is shortened)
        self.assertAlmostEqual(statx.chi_square(a, b), statx.chi_square(b, a))
        # Computation of chi-square p-value (b is shortened)
        aa = statx.chi_square(b[0:(-8)], a)
        bb = statx.chi_square(a, b[0:(-8)])
        self.assertAlmostEqual(aa[0], bb[0])  # p-value
        self.assertAlmostEqual(aa[1], bb[1])  # chi-square value


class SampleSizeTestCases(StatisticsTestCase):
    """
      Test cases for the sample_size() function in core.statistics.
      """

    def test__sample_size__empty_list_numeric(self):
        """
        Empty list returns 0.

        """
        self.assertEqual(statx.sample_size([]), 0)

    def test__sample_size__list_numeric(self):
        """
        Result of sample_size() is number of elements of a list.
        """
        x = [1, 1, 2, 5, 8]
        self.assertEqual(statx.sample_size(x), 5)

    def test__sample_size__nparray_numeric_with_nan(self):
        """
        Result of sample_size() is number of elements of numpy array minus
        number of NaNs.
        """
        x = np.array([1, 1, np.nan, 2, np.nan, 5, 8])
        self.assertEqual(statx.sample_size(x), 5)

    def test__sample_size__list_categorical(self):
        """
        Result of sample_size() is number of elements of a list of categorical
        data.
        """
        x = ['1', '1', '3', '2', '6', '5', '8']
        self.assertEqual(statx.sample_size(x), 7)

    def test__sample_size__pdseries_categorical(self):
        """
        Result of sample_size() is number of elements of a pandas series with
        categorical data.
        """
        x = pd.Series(['1', '7', '2', '5', '8', '0'])
        self.assertEqual(statx.sample_size(x), 6)

    def test__sample_size__pdseries_categorical_with_na(self):
        """
        Result of sample_size() is number of elements of a pandas series with
        categorical data including NAs.
        """
        x = ['1', '1', 'NA', '2', 'NA', '5', '8']
        self.assertEqual(statx.sample_size(x), 5)


class EstimateSampleSizeTestCases(StatisticsTestCase):
    """
    Test cases for the estimate_sample_size() function in core.statistics.
    """

    def test__estimate_sample_size__dataframe(self):
        """
        Result of estimate_sample_size() is a series of estimated sample sizes.
        """
        x = pd.DataFrame(
            {
                'sample_1': [1, 7, 8, 9, 3, 4, 2, 0],
                'sample_2': [5, 4, 3, 2, 2, 2, 1, 0]
            }
        )

        res = statx.estimate_sample_size(x=x, mde=0.01, r=1.0, n=2)

        self.assertEqual(int(res['sample_1']), 197405)
        self.assertEqual(int(res['sample_2']), 142130)

    def test__estimate_sample_size__series(self):
        """
        Result of estimate_sample_size() is estimated sample size.
        """
        x = pd.Series([1, 7, 8, 9, 3, 4, 2, 0])
        self.assertEqual(int(statx.estimate_sample_size(x=x, mde=0.01, r=1.0, n=2)), 197405)

    def test__estimate_sample_size__x_type_error(self):
        """
        Method estimate_sample_size raises TypeError since x is a list.
        """
        x = [1, 7, 8, 9, 3, 4, 2, 0]
        self.assertRaises(TypeError, statx.estimate_sample_size, x=x, mde=0.01, r=1.0, n=2)

    def test__estimate_sample_size__n_type_error(self):
        """
        Method estimate_sample_size raises TypeError since n is a float.
        """
        x = pd.Series([1, 7, 8, 9, 3, 4, 2, 0])
        self.assertRaises(TypeError, statx.estimate_sample_size, x=x, mde=0.01, r=1.0, n=1.5)

    def test__estimate_sample_size__n_value_error(self):
        """
        Method estimate_sample_size raises ValueError since n is 0.
        """
        x = pd.Series([1, 7, 8, 9, 3, 4, 2, 0])
        self.assertRaises(ValueError, statx.estimate_sample_size, x=x, mde=0.01, r=1.0, n=0)

    def test__estimate_sample_size__r_value_error(self):
        """
        Method estimate_sample_size raises ValueError since r is 0.
        """
        x = pd.Series([1, 7, 8, 9, 3, 4, 2, 0])
        self.assertRaises(ValueError, statx.estimate_sample_size, x=x, mde=0.01, r=0.0, n=2)


class AlphaToPercentilesTestCases(StatisticsTestCase):
    """
      Test cases for the alpha_to_percentiles() function in core.statistics.
      """

    def test__alpha_to_percentiles__computation(self):
        """
        Result of alpha_to_percentiles() equals expected result.
        """
        # Computing alpha_to_percentiles
        result = statx.alpha_to_percentiles(0.05)
        # Checking if first percentile is correct
        self.assertEqual(result[0], 2.5)
        # Checking if second percentile is correct
        self.assertEqual(result[1], 97.5)


class BootstrapTestCases(StatisticsTestCase):
    """
      Test cases for the bootstrap() function in core.statistics.
      """
    def test__bootstrap__computation(self):
        """
        Result of bootstrap() equals expected result.
        """
        # Computing bootstrap
        result1 = statx.bootstrap(self.samples.temperature, [0],
                                  min_observations=1)
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

        # Defining data and computing bootstrap
        sample4 = self.samples.temperature[self.samples.gender == 1]
        sample5 = self.samples.temperature[self.samples.gender == 2]
        result4 = statx.bootstrap(sample4, sample5, percentiles=[0.1, 99.9])
        # Checking if lower percentile of result3 is correct
        self.assertAlmostEqual(result4[0][0.1], -0.67078000000000748)
        # Checking if upper percentile of result3 is correct
        self.assertAlmostEqual(result4[0][99.9], 0.093849230769235695)


class PooledStdTestCases(StatisticsTestCase):
    """
      Test cases for the pooled_std() function in core.statistics.
      """

    def test__pooled_std__variances_differ_too_much_error(self):
        """
        Warning raised when variances differ too much.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            statx.pooled_std(0.25, 4, 0.5, 4)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue('variances differ' in str(w[-1].message))

    def test__pooled_std__computation(self):
        """
        Result of pooled_std() equals expected result.
        """
        # Define subset of data for test
        sbp = self.metrics.loc['systolic_bp', :]
        # Computation of pooled standard deviation
        result1 = statx.pooled_std(sbp.men.s, sbp.men.n,
                                   sbp.women.s, sbp.women.n)
        # Checking if result1 is correct
        self.assertAlmostEqual(result1, 19, 1)

        # Computation of pooled standard deviation
        result2 = statx.pooled_std(9.7, 6, 12.0, 4)
        # Checking pooled_std() on small sample subset
        self.assertAlmostEqual(result2, 10.6, 1)


class NormalSamplePercentilesTestCases(StatisticsTestCase):
    """
      Test cases for the normal_sample_percentiles() function in core.statistics.
      """

    def test__normal_sample_percentiles__computation(self):
        """
        Result of normal_sample_percentiles() equals expected result.

        Example from:
        http://www.stat.yale.edu/Courses/1997-98/101/confint.htm
        """
        # Define data for testing
        val = np.array([102.5, 101.7, 103.1, 100.9, 100.5, 102.2])
        # Computation with relative set True
        result1 = statx.normal_sample_percentiles(val, relative=False)
        # Checking if lower percentile of result1 is correct
        self.assertAlmostEqual(result1[2.5], 100.87330892117532)
        # Checking if upper percentile of result1 is correct
        self.assertAlmostEqual(result1[97.5], 102.760024412158)

        # Computation with relative set True
        result2 = statx.normal_sample_percentiles(val, relative=True)
        # Checking if lower percentile of result2 is correct
        self.assertAlmostEqual(result2[2.5], -0.94335774549134732)
        # Checking if upper percentile of result2 is correct
        self.assertAlmostEqual(result2[97.5], 0.94335774549134732)


class NormalDifferenceTestCases(StatisticsTestCase):
    """
      Test cases for the normal_difference() function in core.statistics.
      """

    def test__normal_difference__computation(self):
        """
        Result of normal_difference() equals expected result.
        """
        # Define subset of data for first test
        sbp = self.metrics.loc['systolic_bp', :]
        # computation of normal difference
        result1 = statx.normal_difference(sbp.men.m, sbp.men.s, sbp.men.n,
                                          sbp.women.m, sbp.women.s, sbp.women.n)
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
    """
      Test cases for the normal_sample_difference() function in core.statistics.
      """

    def test__normal_sample_difference__computation(self):
        # Defining data and computing normal difference
        sample1 = self.samples.temperature[self.samples.gender == 1]
        sample2 = self.samples.temperature[self.samples.gender == 2]
        # Computation of normal sample difference
        result1 = statx.normal_sample_difference(sample1, sample2)
        # Checking if lower percentile of result1 is correct
        self.assertAlmostEqual(result1[2.5], -0.53770569567692295)
        # Checking if upper percentile of result1 is correct
        self.assertAlmostEqual(result1[97.5], -0.040755842784587965)


class StatisticalPowerTestCases(StatisticsTestCase):
    """
      Test cases for compute_statistical_power() function in core.statistics.
      """

    def test_compute_statistical_power(self):
        # pre-computed value by hand via power analysis
        # alpha=0.05, beta=0.2, sigma=1, delta=1, n1=12, n2=13
        z_1_minus_alpha = stats.norm.ppf(0.95)
        power = statx._get_power(0, 1, 13, 1, 1, 12, z_1_minus_alpha)
        self.assertAlmostEqual(power, 0.8, 2)


if __name__ == '__main__':
    unittest.main()
