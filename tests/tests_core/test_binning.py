# TODO: This module is deprecated

import sys
import unittest

import pandas as pd

from expan.core.binning import *

#---------- test util function ------------#
def toBinRepresentation(bins):
    """
    Get the bin representations of the given bins
    :param bins: a list of Bin objects.
    :return: a lista list of NumericalRepresenation or CategoricalRepresenation objects
    """
    reprs = []
    for bin in bins:
        reprs.append(bin.representation)
    return reprs


#---------- test classes ------------#
class BinningTestCase(unittest.TestCase):
    """
    Defines the setUp() and tearDown() functions for the statistics test cases.
    """
    def setUp(self):
        np.random.seed(41)
        self.maxDiff = None

    def tearDown(self):
        pass

    def assertCollectionEqual(self, source, expected):
        is_python3 = sys.version_info[0] == 3
        if is_python3:
            return self.assertCountEqual(source, expected)
        else:
            return self.assertItemsEqual(source, expected)



#---------- Numerical binning tests ------------#
class CreateNumericalBinsTestCase(BinningTestCase):
    """
    Test cases for creating numerical bins.
    """
    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            create_bins(None, 10)
        with self.assertRaises(ValueError):
            create_bins(list(range(100)), 0)

    def test_create_regular(self):
        data = np.arange(1000)
        nbins = 10
        bins = create_bins(data, nbins)
        self.assertEqual(len(bins), nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresentation(0, 100, True, False),
            NumericalRepresentation(100, 200, True, False),
            NumericalRepresentation(200, 300, True, False),
            NumericalRepresentation(300, 400, True, False),
            NumericalRepresentation(400, 500, True, False),
            NumericalRepresentation(500, 600, True, False),
            NumericalRepresentation(600, 700, True, False),
            NumericalRepresentation(700, 800, True, False),
            NumericalRepresentation(800, 900, True, False),
            NumericalRepresentation(900, 999, True, True)
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_create_nan(self):
        data = np.arange(1002.)
        data[-2] = np.nan
        data[-1] = np.nan
        nbins = 11
        bins = create_bins(data, nbins)
        self.assertEqual(len(bins), nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresentation(0, 100, True, False),
            NumericalRepresentation(100, 200, True, False),
            NumericalRepresentation(200, 300, True, False),
            NumericalRepresentation(300, 400, True, False),
            NumericalRepresentation(400, 500, True, False),
            NumericalRepresentation(500, 600, True, False),
            NumericalRepresentation(600, 700, True, False),
            NumericalRepresentation(700, 800, True, False),
            NumericalRepresentation(800, 900, True, False),
            NumericalRepresentation(900, 999, True, True),
            NumericalRepresentation(np.nan, np.nan, True, True)  # bin for nans
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_creation_two_big_bins_noise_between(self):
        data = [0] * 10000 + list(range(300)) + [301] * 10000
        nbins = 10
        with warnings.catch_warnings(record=True) as w:
            bins = create_bins(data, nbins)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue('less bins than requested' in str(w[-1].message).lower())
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresentation(0, 0, True, True),
            NumericalRepresentation(1, 301, True, False),
            NumericalRepresentation(301, 301, True, True)
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_creation_very_skewed_data(self):
        data = [0] * 10000 + list(range(300))
        nbins = 4
        bins = create_bins(data, nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresentation(0, 0, True, True),
            NumericalRepresentation(1, 101, True, False),
            NumericalRepresentation(101, 200, True, False),
            NumericalRepresentation(200, 299, True, True)
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_creation_single_bin(self):
        data = [0] * 100
        nbins = 1
        bins = create_bins(data, nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [NumericalRepresentation(0, 0, True, True)]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_creation_more_bins_than_data(self):
        data = [0] * 100 + [1] * 10
        nbins = 3
        with warnings.catch_warnings(record=True) as w:
            bins = create_bins(data, nbins)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue('unique values' in str(w[-1].message))
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresentation(0, 0, True, True),
            NumericalRepresentation(1, 1, True, True)
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_creation_range_100__n_bins_8(self):
        data = list(range(100))
        nbins = 8
        bins = create_bins(data, nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresentation(0, 13, True, False),
            NumericalRepresentation(13, 26, True, False),
            NumericalRepresentation(26, 39, True, False),
            NumericalRepresentation(39, 51, True, False),
            NumericalRepresentation(51, 63, True, False),
            NumericalRepresentation(63, 76, True, False),
            NumericalRepresentation(76, 88, True, False),
            NumericalRepresentation(88, 99, True, True)
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_creation_range_100__n_bins_2(self):
        data = list(range(100))
        nbins = 2
        bins = create_bins(data, nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresentation(0, 50, True, False),
            NumericalRepresentation(50, 99, True, True)
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)


class ApplyNumericalBinsTestCase(BinningTestCase):
    """
    Test cases for applying bins to numerical data.
    """
    def test_assign_regular(self):
        data = pd.DataFrame(np.tile(np.array([np.arange(1000)]).T, (1,3)), columns=list('ABC'))
        dimension = 'A'

        bin1 = Bin("numerical", 0, 10, True, False)
        data_applied_bin1 = pd.DataFrame(np.tile(np.array([np.arange(10)]).T, (1, 3)), columns=list('ABC'))
        np.testing.assert_array_equal(data_applied_bin1, bin1(data, dimension))

        bin2 = Bin("numerical", 40, 50, True, False)
        data_applied_bin2 = pd.DataFrame(np.tile(np.array([np.arange(40, 50)]).T, (1, 3)), columns=list('ABC'))
        np.testing.assert_array_equal(data_applied_bin2, bin2(data, dimension))

        bin3 = Bin("numerical", 300, 500, False, True)
        data_applied_bin3 = pd.DataFrame(np.tile(np.array([np.arange(301, 501)]).T, (1, 3)), columns=list('ABC'))
        np.testing.assert_array_equal(data_applied_bin3, bin3(data, dimension))

        bin4 = Bin("numerical", 900, 999, True, True)
        data_applied_bin4 = pd.DataFrame(np.tile(np.array([np.arange(900, 1000)]).T, (1, 3)), columns=list('ABC'))
        np.testing.assert_array_equal(data_applied_bin4, bin4(data, dimension))

    def test_assign_unseen_data(self):
        data = pd.DataFrame(np.tile(np.array([np.arange(1000)]).T, (1,3)), columns=list('ABC'))
        dimension = 'A'

        bin = Bin("numerical", 1000, 2000, False, False)
        np.testing.assert_array_equal(None, bin(data, dimension))

    def test_assign_nan(self):
        data_one_dim = np.arange(1002.)
        data_one_dim[-2] = np.nan
        data_one_dim[-1] = np.nan
        data = pd.DataFrame(np.tile(np.array([data_one_dim]).T, (1,3)), columns=list('ABC'))
        dimension = 'A'

        bin_regular = Bin("numerical", 0, 10, True, False)
        data_applied_bin_regular = pd.DataFrame(np.tile(np.array([np.arange(10)]).T, (1, 3)), columns=list('ABC'))
        np.testing.assert_array_equal(data_applied_bin_regular, bin_regular(data, dimension))

        bin_nan = Bin("numerical", np.nan, np.nan, True, True)
        data_applied_bin_nan = pd.DataFrame( np.full((2,3), np.nan), columns=list('ABC'))
        np.testing.assert_array_equal(data_applied_bin_nan, bin_nan(data, dimension))


#---------- Categorical binning tests ------------#
class CreateCategoricalBinsTestCase(BinningTestCase):
    """
    Test cases for creating categorical bins.
    """
    def test_categorical_binning_1(self):
        data = ['a'] * 10 + ['b'] * 10
        bins = create_bins(data, 2)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            CategoricalRepresentation(["a"]),
            CategoricalRepresentation(["b"])
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_categorical_binning_2(self):
        data = ['a'] * 10 + ['b'] * 5 + ['c'] * 5
        bins = create_bins(data, 2)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            CategoricalRepresentation(["a"]),
            CategoricalRepresentation(["c", "b"])
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)

    def test_categorical_binning_too_little_data(self):
        data = ['a'] * 10 + ['b'] * 5 + ['c'] * 5
        bins = create_bins(data, 4)
        self.assertEqual(len(bins), 3)

    def test_binning_date_1(self):
        data = ['2017-05-01'] * 10 + ['2017-06-01'] * 5 + ['2017-07-01'] * 5

        bins = create_bins(data, 4)
        self.assertEqual(len(bins), 3)

    def test_binning_date_2(self):
        data = ['2017-05-01'] * 10 + ['2017-06-01'] * 5 + ['2017-07-01'] * 5

        bins = create_bins(data, 2)
        self.assertEqual(len(bins), 2)

        bins_repr_source = toBinRepresentation(bins)

        bins_repr_expected = [
            CategoricalRepresentation(["2017-05-01"]),
            CategoricalRepresentation(["2017-07-01", "2017-06-01"])
        ]
        self.assertCollectionEqual(bins_repr_source, bins_repr_expected)


class ApplyCategoricalBinsTestCase(BinningTestCase):
    """
    Test cases for applying bins to categorical data.
    """
    def test_assign_regular(self):
        data = pd.DataFrame(["a", "b", "c", "a", "b"], columns=list("a"))

        bin_a = Bin("categorical", ["a"])
        np.testing.assert_array_equal(np.array([["a", "a"]]).T, bin_a(data, "a"))

        bin_b = Bin("categorical", ["b"])
        np.testing.assert_array_equal(np.array([["b", "b"]]).T, bin_b(data, "a"))

        bin_c = Bin("categorical", ["c"])
        np.testing.assert_array_equal(np.array([["c"]]), bin_c(data, "a"))

    def test_assign_unseen(self):
        data = pd.DataFrame(["a", "b", "c"], columns=list("a"))

        bin = Bin("categorical", ["d"])
        np.testing.assert_array_equal(None, bin(data, "a"))

    def test_assign_multiple(self):
        data = pd.DataFrame(["a", "b", "c", "a", "b"], columns=list("a"))

        bin = Bin("categorical", ["a", "b"])
        np.testing.assert_array_equal(np.array([["a", "b", "a", "b"]]).T, bin(data, "a"))
