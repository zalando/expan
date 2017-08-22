import unittest

from expan.core.binning import *


class BinningTestCase(unittest.TestCase):
    """
    Defines the setUp() and tearDown() functions for the statistics test cases.
    """

    def setUp(self):
        np.random.seed(41)

    def tearDown(self):
        pass


class CreateNumericalBinsTestCase(BinningTestCase):
    """
    Test cases for creating numerical bins.
    """
    def test_bin_representation(self):
        pass

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            create_bins(None, 10)

        with self.assertRaises(ValueError):
            create_bins(list(range(100)), 0)

    def test_create_numerical_bin(self):
        data = np.arange(1000)
        nbins = 10

        bins = create_bins(data, nbins)
        self.assertEqual(len(bins), nbins)


class AssignNumericalBinsTestCase(BinningTestCase):
    """
    Test cases for assigning bins to numerical data.
    """
    def test_assign_numerical_bins(self):
        pass


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
