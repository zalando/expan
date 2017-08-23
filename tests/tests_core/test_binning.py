import unittest

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
            NumericalRepresenation(0, 100, True, False),
            NumericalRepresenation(100, 200, True, False),
            NumericalRepresenation(200, 300, True, False),
            NumericalRepresenation(300, 400, True, False),
            NumericalRepresenation(400, 500, True, False),
            NumericalRepresenation(500, 600, True, False),
            NumericalRepresenation(600, 700, True, False),
            NumericalRepresenation(700, 800, True, False),
            NumericalRepresenation(800, 900, True, False),
            NumericalRepresenation(900, 999, True, True)
        ]
        self.assertCountEqual(bins_repr_source, bins_repr_expected)

    def test_create_nan(self):
        data = np.arange(1002.)
        data[-2] = np.nan
        data[-1] = np.nan
        nbins = 11
        bins = create_bins(data, nbins)
        self.assertEqual(len(bins), nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresenation(0, 100, True, False),
            NumericalRepresenation(100, 200, True, False),
            NumericalRepresenation(200, 300, True, False),
            NumericalRepresenation(300, 400, True, False),
            NumericalRepresenation(400, 500, True, False),
            NumericalRepresenation(500, 600, True, False),
            NumericalRepresenation(600, 700, True, False),
            NumericalRepresenation(700, 800, True, False),
            NumericalRepresenation(800, 900, True, False),
            NumericalRepresenation(900, 999, True, True),
            NumericalRepresenation(np.nan, np.nan, True, True)  # bin for nans
        ]
        self.assertCountEqual(bins_repr_source, bins_repr_expected)

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
            NumericalRepresenation(0, 0, True, True),
            NumericalRepresenation(1.0, 301.0, True, False),
            NumericalRepresenation(301.0, 301.0, True, True)
        ]
        self.assertCountEqual(bins_repr_source, bins_repr_expected)

    def test_creation_very_skewed_data(self):
        data = [0] * 10000 + list(range(300))
        nbins = 4
        bins = create_bins(data, nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresenation(0.0, 0.0, True, True),
            NumericalRepresenation(1.0, 101.0, True, False),
            NumericalRepresenation(101.0, 200.0, True, False),
            NumericalRepresenation(200.0, 299.0, True, True)
        ]
        self.assertCountEqual(bins_repr_source, bins_repr_expected)

    def test_creation_single_bin(self):
        data = [0] * 100
        nbins = 1
        bins = create_bins(data, nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [NumericalRepresenation(0.0, 0.0, True, True)]
        self.assertCountEqual(bins_repr_source, bins_repr_expected)

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
            NumericalRepresenation(0.0, 0.0, True, True),
            NumericalRepresenation(1.0, 1.0, True, True)
        ]
        self.assertCountEqual(bins_repr_source, bins_repr_expected)

    def test_creation_range_100__n_bins_8(self):
        data = list(range(100))
        nbins = 8
        bins = create_bins(data, nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresenation(0, 13, True, False),
            NumericalRepresenation(13, 26, True, False),
            NumericalRepresenation(26, 39, True, False),
            NumericalRepresenation(39, 51, True, False),
            NumericalRepresenation(51, 63, True, False),
            NumericalRepresenation(63, 76, True, False),
            NumericalRepresenation(76, 88, True, False),
            NumericalRepresenation(88, 99, True, True)
        ]
        self.assertCountEqual(bins_repr_source, bins_repr_expected)

    def test_creation_range_100__n_bins_2(self):
        data = list(range(100))
        nbins = 2
        bins = create_bins(data, nbins)
        bins_repr_source = toBinRepresentation(bins)
        bins_repr_expected = [
            NumericalRepresenation(0, 50, True, False),
            NumericalRepresenation(50, 99, True, True)
        ]
        self.assertCountEqual(bins_repr_source, bins_repr_expected)


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
        self.assertEqual(labels[0].representation, NumericalRepresenation(0, 100, True, False))
        self.assertEqual(labels[100].representation, NumericalRepresenation(100, 200, True, False))
        self.assertEqual(labels[200].representation, NumericalRepresenation(200, 300, True, False))
        self.assertEqual(labels[450].representation, NumericalRepresenation(400, 500, True, False))
        self.assertEqual(labels[799].representation, NumericalRepresenation(700, 800, True, False))
        self.assertEqual(labels[800].representation, NumericalRepresenation(800, 900, True, False))
        self.assertEqual(labels[999].representation, NumericalRepresenation(900, 999, True, True))

    def test_assign_unseen_data(self):
        seen = np.arange(1000)
        nbins = 10
        unseen = np.arange(1100)

        bins = create_bins(seen, nbins)
        self.assertEqual(len(bins), nbins)

        labels = assign_bins(unseen, bins)
        self.assertEqual(labels[0].representation, NumericalRepresenation(0, 100, True, False))
        self.assertEqual(labels[100].representation, NumericalRepresenation(100, 200, True, False))
        self.assertEqual(labels[200].representation, NumericalRepresenation(200, 300, True, False))
        self.assertEqual(labels[450].representation, NumericalRepresenation(400, 500, True, False))
        self.assertEqual(labels[799].representation, NumericalRepresenation(700, 800, True, False))
        self.assertEqual(labels[800].representation, NumericalRepresenation(800, 900, True, False))
        self.assertEqual(labels[999].representation, NumericalRepresenation(900, 999, True, True))
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
            self.assertEqual(len(w), 2)

        self.assertEqual(labels[1].representation, NumericalRepresenation(0, 100, True, False))
        self.assertEqual(labels[100].representation, NumericalRepresenation(100, 200, True, False))
        self.assertEqual(labels[200].representation, NumericalRepresenation(200, 300, True, False))
        self.assertEqual(labels[450].representation, NumericalRepresenation(400, 500, True, False))
        self.assertEqual(labels[799].representation, NumericalRepresenation(700, 800, True, False))
        self.assertEqual(labels[800].representation, NumericalRepresenation(800, 900, True, False))
        self.assertEqual(labels[999].representation, NumericalRepresenation(900, 999, True, True))
        self.assertEqual(labels[0], None)
        self.assertEqual(labels[1000].representation, NumericalRepresenation(np.nan, np.nan, True, True))
        self.assertEqual(labels[1001].representation, NumericalRepresenation(np.nan, np.nan, True, True))


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
