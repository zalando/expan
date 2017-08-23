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
        bins_repr_source = toBinRepresentation(bins)
        print(bins_repr_source)
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


class AssignNumericalBinsTestCase(BinningTestCase):
    """
    Test cases for assigning bins to numerical data.
    """
    def test_assign_numerical_bins(self):
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



