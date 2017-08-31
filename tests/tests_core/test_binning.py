import unittest

import sys

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


class AssignNumericalBinsTestCase(BinningTestCase):
    """
    Test cases for assigning bins to numerical data.
    """
    def test_assign_regular(self):
        data = np.arange(1000)
        nbins = 10

        bins = create_bins(data, nbins)
        self.assertEqual(len(bins), nbins)

        labels = assign_bins(data, bins)
        self.assertEqual(labels[0].representation, NumericalRepresentation(0, 100, True, False))
        self.assertEqual(labels[100].representation, NumericalRepresentation(100, 200, True, False))
        self.assertEqual(labels[200].representation, NumericalRepresentation(200, 300, True, False))
        self.assertEqual(labels[450].representation, NumericalRepresentation(400, 500, True, False))
        self.assertEqual(labels[799].representation, NumericalRepresentation(700, 800, True, False))
        self.assertEqual(labels[800].representation, NumericalRepresentation(800, 900, True, False))
        self.assertEqual(labels[999].representation, NumericalRepresentation(900, 999, True, True))

    def test_assign_unseen_data(self):
        seen = np.arange(1000)
        nbins = 10
        unseen = np.arange(1100)

        bins = create_bins(seen, nbins)
        self.assertEqual(len(bins), nbins)

        labels = assign_bins(unseen, bins)
        self.assertEqual(labels[0].representation, NumericalRepresentation(0, 100, True, False))
        self.assertEqual(labels[100].representation, NumericalRepresentation(100, 200, True, False))
        self.assertEqual(labels[200].representation, NumericalRepresentation(200, 300, True, False))
        self.assertEqual(labels[450].representation, NumericalRepresentation(400, 500, True, False))
        self.assertEqual(labels[799].representation, NumericalRepresentation(700, 800, True, False))
        self.assertEqual(labels[800].representation, NumericalRepresentation(800, 900, True, False))
        self.assertEqual(labels[999].representation, NumericalRepresentation(900, 999, True, True))
        self.assertEqual(labels[1000], None)
        self.assertEqual(labels[1050], None)
        self.assertEqual(labels[1099], None)

    def test_assign_nan(self):
        data = np.arange(1002.)
        data[-2] = np.nan
        data[-1] = np.nan
        nbins = 11

        bins = create_bins(data, nbins)
        self.assertEqual(len(bins), nbins)

        data[0] = 2000
        with warnings.catch_warnings(record=True) as w:
            labels = assign_bins(data, bins)
            self.assertEqual(len(w), 3)

        self.assertEqual(labels[1].representation, NumericalRepresentation(0, 100, True, False))
        self.assertEqual(labels[100].representation, NumericalRepresentation(100, 200, True, False))
        self.assertEqual(labels[200].representation, NumericalRepresentation(200, 300, True, False))
        self.assertEqual(labels[450].representation, NumericalRepresentation(400, 500, True, False))
        self.assertEqual(labels[799].representation, NumericalRepresentation(700, 800, True, False))
        self.assertEqual(labels[800].representation, NumericalRepresentation(800, 900, True, False))
        self.assertEqual(labels[999].representation, NumericalRepresentation(900, 999, True, True))
        self.assertEqual(labels[0], None)
        self.assertEqual(labels[1000].representation, NumericalRepresentation(np.nan, np.nan, True, True))
        self.assertEqual(labels[1001].representation, NumericalRepresentation(np.nan, np.nan, True, True))


#---------- Categorical binning tests ------------#
class CreateCategoricalBinsTestCase(BinningTestCase):
    """
    Test cases for creating categorical bins.
    """
    pass


class AssignCategoricalBinsTestCase(BinningTestCase):
    """
    Test cases for assigning bins to categorical data.
    """
    pass
