import unittest
import warnings

# The simplefilter(always) stupidly does not clear the existing registry for the
# warnings, meaning that if any particular warning has already been masked by
# the 'once' filter, it will continue to be masked
# see https://docs.python.org/2/library/warnings.html#testing-warnings
warnings.simplefilter('always')

from expan.core.binning import *


class UtilTestCase(unittest.TestCase):
	"""
	Defines the setUp() and tearDown() functions for the statistics test cases.
	"""

	def setUp(self):
		"""
		Load the needed datasets for all UtilTestCases and set the random
		seed so that randomized algorithms show deterministic behaviour.
		"""
		np.random.seed(0)

	def tearDown(self):
		"""
		Clean up after the test
		"""
		# TODO: find out if we have to remove data manually
		pass


class BinningTestCase(UtilTestCase):
	"""
	Test cases for the binning function in core.
	"""

	def test_invalid_input(self):
		"""
		Check if errors are raised for invalid input (empty data and n_bins=0)
		"""

		# Check if error is raised for empty input
		with self.assertRaises(ValueError):
			create_binning(None, 10)

		with self.assertRaises(ValueError):
			create_binning(range(100), 0)

	def test_creation_range_100__nbins_1(self):
		"""
		Check if n_bins = 1 functions properly.
		"""
		# Create data
		x = range(100)
		# Calculate binning
		bins = create_binning(x, 1)
		r = bins.label(x, '{simplei}')
		# Expected result
		e = ['0_99']
		# Comparison
		self.assertEqual(r[0:1], e)

	def test_creation_range_100__nbins_2(self):
		"""
		Check if n_bins = 2 functions properly.
		"""
		# Create data
		x = range(100)
		# Calculate binning
		bins = create_binning(x, 2)
		r = bins.label(x, '{simplei}')
		# Expected result
		self.assertEqual(r[0], '0_50')
		self.assertEqual(r[-1], '50_99')

	def test_creation_range_100__n_bins_5(self):
		"""
		Check if n_bins = 5 functions properly.
		"""
		# Create data
		x = range(100)
		# Calculate binning
		bins = create_binning(x, 5)
		r = bins.label(x, '{simplei}')
		# Expected result
		# Comparison
		self.assertEqual(r[0], '0_20')
		self.assertEqual(r[20], '20_40')
		self.assertEqual(r[40], '40_60')
		self.assertEqual(r[60], '60_80')
		self.assertEqual(r[80], '80_99')
		self.assertEqual(r[99], '80_99')

	def test_creation_range_100__n_bins_8(self):
		"""
		Check if n_bins = 8 functions properly.
		"""
		# Create data
		x = range(100)
		# Calculate binning
		bins = create_binning(x, 8)
		labels = bins.label(x, '{simplei}')
		r = list(bins.labels('{simplei}'))
		r.sort()
		# Expected result
		e = ['88_99', '0_13', '63_76', '39_51', '51_63', '76_88', '26_39', '13_26']
		e.sort()
		# Comparison
		self.assertEqual(r, e)

	def test_creation_range_100__n_bins_10(self):
		"""
		Check if n_bins = 10 functions properly.
		"""
		# Create data
		x = range(100)
		# Calculate binning
		bins = create_binning(x, 10)
		r = bins.label(x, '{simplei}')

		# Expected result
		self.assertEqual(r[0], '0_10')
		self.assertEqual(r[10], '10_20')
		self.assertEqual(r[-1], '90_99')

	def test_creation_range_100__n_bins_20(self):
		"""
		Check if n_bins = 20 functions properly.
		"""
		# Create data
		x = range(100)

		# Calculate binning
		bins = create_binning(x, 20)
		r = bins.label(x, '{simplei}')

		# Expected result
		self.assertEqual(r[0], '0_5')
		self.assertEqual(r[6], '5_10')
		self.assertEqual(r[-1], '95_99')

	def test_creation_more_bins_than_data(self):
		"Check warnings and result if data has insufficient values for bins"
		# Create data
		x = [0] * 100 + [1] * 10
		# Calculate binning
		w = None
		with warnings.catch_warnings(record=True) as w:
			bins = create_binning(x, 3)

			self.assertEqual(len(w), 1)
			self.assertTrue(issubclass(w[-1].category, UserWarning))
			self.assertTrue('distinct values' in str(w[-1].message))

		r = list(bins.labels('{standard}'))
		r.sort()
		# Expected result
		e = ['[0.0,0.0]', '[1.0,1.0]']
		e.sort()
		# Comparison
		self.assertEqual(r, e)

	def test_creation_single_bin(self):
		"Check result if data has single value and request single bin"
		# Create data
		x = [0] * 100
		# Calculate binning
		bins = create_binning(x, 1)
		r = list(bins.labels())
		r.sort()
		# Expected result
		e = ['[0.0,0.0]']
		e.sort()
		# Comparison
		self.assertEqual(r, e)

	def test_creation_very_skewed_data(self):
		# Create data
		x = [0] * 10000 + range(300)
		# Calculate binning
		with warnings.catch_warnings(record=True) as w:
			bins = create_binning(x=x, nbins=4)
		r = bins.labels().tolist()
		# r.sort()
		# Expected result
		# e = ['0_1','1_100','100_200','200_300'] # DEFINE EXPECTED RESULT
		e = ['[0.0,0.0]', '[1.0,101.0)', '[101.0,200.0)', '[200.0,299.0]']
		# e.sort()
		# Comparison
		self.assertEqual(r, e)

	# How about this:
	# zero-inflated distribution
	# >>> dat = np.concatenate((np.random.poisson(3,1000), np.zeros(1000), np.repeat(np.nan,100)))

	# >>> interval_dict = get_binning(dat, 5)
	# >>> len(interval_dict)
	# 6

	# >>> dat = np.random.choice(['a','b','c','d','e'], 1000)
	# >>> len(get_binning(dat))
	# 5
	# >>> print set_binning(dat, interval_dict)
	# ['0.0_2.0' '2.0_3.0' '3.0_4.0' ..., '0.0_2.0' '0.0_2.0' '0.0_2.0']

	def test_creation_two_big_bins_noise_between(self):
		# Create data
		x = [0] * 10000 + range(300) + [301] * 10000
		# Calculate binning

		with warnings.catch_warnings(record=True) as w:
			bins = create_binning(x=x, nbins=10)
			self.assertEqual(len(w), 1)
			self.assertTrue(issubclass(w[-1].category, UserWarning))
			self.assertTrue('less bins than requested' in str(w[-1].message).lower())
		r = bins.labels().tolist()
		# r.sort()
		# Expected result
		# e = ['0_1','1_100','100_200','200_300','301_301'] # DEFINE EXPECTED RESULT
		e = ['[0.0,0.0]', '[1.0,301.0)', '[301.0,301.0]']
		# e.sort()
		# Comparison
		self.assertEqual(r, e)


class NumericalBinningClassTestCase(UtilTestCase):
	def test_manual_creation(self):
		"Test manual creation of numerical binning"
		bb = NumericalBinning()

		bb.lowers = [0, 1, 2, 3]
		bb.uppers = [1, 2, 3, 4]
		bb.lo_closed = [True] * 4
		bb.up_closed = [False] * 4
		bb.up_closed[3] = True

		n = np.nan

		with warnings.catch_warnings(record=True) as w:
			# ignore the invalid value warnings from the nan
			warnings.simplefilter('ignore', category=RuntimeWarning)
			res = bb._apply([-1, 0, 1, 2, 3, 4, 5, n, 1.99])

		np.testing.assert_array_equal(res, [-1, 0, 1, 2, 3, 3, -1, -1, 1])

	def test_manual_nan_bin(self):
		"Test handling of nan bin in numerical binning (manual creation)"
		bins = NumericalBinning()

		n = np.nan
		bins.lowers = [0, 1, 2, n]
		bins.uppers = [1, 2, 3, n]
		bins.lo_closed = [True] * 4
		bins.up_closed = [False] * 4

		# with warnings.catch_warnings(record=True) as w:
		# ignore the invalid value warnings from the nan
		#	warnings.simplefilter('ignore', category=RuntimeWarning)

		res = bins._apply([-1, 0, 1, 2, 3, n, 1.99, 2.99999])
		np.testing.assert_array_equal(res, [-1, 0, 1, 2, -1, 3, 1, 2])

	def test_binning(self):
		"Test various functions of numerical binning"
		values = np.arange(1000)
		nbins = 10

		bins = create_binning(values, nbins)
		self.assertTrue(isinstance(bins, NumericalBinning))
		self.assertEqual(len(bins), nbins)

		# Test using midpoint as label
		# This allows the label to be passed back into the binning.label
		# and will label identically
		labels = bins.mid(values)
		self.assertEqual(labels[0], 50.)
		self.assertEqual(labels[99], 50.)
		self.assertEqual(labels[100], 150.)

		# Test using midpoint as a formatstr
		labels = bins.label(values, '{mid:.2f}')
		self.assertEqual(labels[0], '50.00')
		self.assertEqual(labels[99], '50.00')

		labels = bins.label(values, '{set_notation}')
		self.assertEqual(labels[0], '[0.0,100.0)')
		self.assertEqual(labels[-1], '[900.0,999.0]')

		# Test a certain label format...
		labels = bins.label(values, '({lo:.0f}{lo_cond}x{up_cond}{up:.0f})')
		self.assertEqual(labels[0], '(0<=x<100)')
		self.assertEqual(labels[-1], '(900<=x<=999)')

		# Test a certain label format with the shortcuts...
		labels = bins.label(values, '({conditions})')
		self.assertEqual(labels[0], '(0.0<=x<100.0)')
		self.assertEqual(labels[-1], '(900.0<=x<=999.0)')

		# See if we can format the numbers how we like
		labels = bins.label(values, '({lo:.2f}{lo_cond}x{up_cond}{up:.2f})')
		self.assertEqual(labels[0], '(0.00<=x<100.00)')
		self.assertEqual(labels[-1], '(900.00<=x<=999.00)')

		# Test a certain label format...
		labels = bins.label(values, '{iter.uppercase}{lo_bracket}{lo:.0f},{up:.0f}{up_bracket}')
		self.assertEqual(len(bins), nbins)
		self.assertEqual(labels[0], 'A[0,100)')
		self.assertEqual(labels[99], 'A[0,100)')
		self.assertEqual(labels[100], 'B[100,200)')
		self.assertEqual(labels[-1], 'J[900,999]')

	def test_unseen_data(self):
		"Test unseen data with numerical binning"
		seen = np.arange(1000.)
		nbins = 10
		unseen = np.arange(1100)

		bins = create_binning(seen, nbins)
		self.assertTrue(isinstance(bins, NumericalBinning))
		self.assertEqual(len(bins), nbins)

		labels = bins.mid(unseen)
		self.assertEqual(labels[0], 50.)
		self.assertEqual(labels[99], 50.)
		self.assertEqual(labels[100], 150.)
		self.assertEqual(labels[1000 - 1], 949.5)  # last bin is closed-closed

		# Test that the unseen data is given NaN
		np.testing.assert_equal(labels[1000], np.nan)  # tests equal

		labels = bins.label(unseen, '({conditions})')
		self.assertEqual(labels[0], '(0.0<=x<100.0)')
		self.assertEqual(labels[-1], '(unseen)')
		# This would be cool: (would require knowing that bins span space)
		# self.assertEqual(labels[-1], '(x<0.0 or x>999.0)')

		labels = bins.label(unseen)
		self.assertEqual(labels[-1], '[unseen]')

		labels = bins.label(unseen, '{iter.uppercase} ({conditions})')
		self.assertEqual(labels[0], 'A (0.0<=x<100.0)')
		self.assertEqual(labels[100], 'B (100.0<=x<200.0)')
		self.assertEqual(labels[-1], '? (unseen)')

		labels = bins.label(unseen, '{iter.uppercase}')
		self.assertEqual(labels[0], 'A')
		self.assertEqual(labels[100], 'B')
		self.assertEqual(labels[-1], '?')

	def test_numerical_nans(self):
		"Test handling of nan in numerical binning"
		n = np.nan
		values = np.arange(1002.)
		values[-2] = n
		values[-1] = n
		nbins = 10

		bins = create_binning(values, nbins)
		self.assertTrue(isinstance(bins, NumericalBinning))

		# we say that nan bins are e
		self.assertEqual(len(bins), nbins + 1)

		labels = bins.label(values)
		self.assertEqual(labels[-2], '[nan,nan]')
		self.assertEqual(labels[-1], '[nan,nan]')
		self.assertEqual(labels[-3], '[900.0,999.0]')
		self.assertEqual(labels[0], '[0.0,100.0)')

		# Test using midpoint as label
		# This allows the label to be passed back into the binning.label
		# and will label identically
		labels = bins.mid(values)
		self.assertTrue(np.isnan(labels[-2]))
		self.assertTrue(np.isnan(labels[-1]))
		self.assertEqual(labels[-3], 949.5)
		self.assertEqual(labels[0], 50.)
		self.assertEqual(labels[99], 50.)
		self.assertEqual(labels[100], 150.)

		# test unseen
		values[-3] = 1001.
		labels = bins.label(values, format_str='{iter.integer} ({conditions})')
		self.assertEqual(labels[-3], '? (unseen)')

		# following assumes the nan bin will be the first bin...
		# this may be an assumption we want to change.
		self.assertEqual(labels[0], '1 (0.0<=x<100.0)')


class CategoricalBinningClassTestCase(UtilTestCase):
	def test_manual_creation(self):
		"Test manual creation of categorical binning"
		bb = CategoricalBinning()
		n = np.nan

		bb.categories += [['a', 'b']]
		bb.categories += [['c', 'd', 'e']]
		bb.categories += [['f', 'g']]
		bb.categories += [['h']]

		res = bb._apply(['z', 'a', 'c', 'f', 'h', n, 'x', 0, 'aa'])

		np.testing.assert_array_equal(res, [-1, 0, 1, 2, 3, -1, -1, -1, -1])

	def test_manual_nan_bin(self):
		"Test handling of nan bin in categorical binning (manual creation)"
		bins = CategoricalBinning()
		n = np.nan

		bins.categories += [['a', 'b']]
		bins.categories += [['c', 'd', 'e']]
		bins.categories += [['f', 'g']]
		bins.categories += [['h', np.nan]]
		bins.categories += [['i', 'j']]

		res = bins._apply(['z', 'a', 'c', 'f', 'h', n, 'x', 0, 'aa', 'j'])
		np.testing.assert_array_equal(res, [-1, 0, 1, 2, 3, 3, -1, -1, -1, 4])

		bins = CategoricalBinning()
		bins.categories += [['a', 'b']]
		bins.categories += [[n]]
		res = bins._apply(['a', 'b', 'c', n])
		np.testing.assert_array_equal(res, [0, 0, -1, 1])

		x = ['A'] * 50 + ['B'] * 10 + ['C'] * 20 + [np.nan] * 10
		bins = create_binning(x=x, nbins=3)
		r = bins.labels().tolist()
		e = ['{A}', '{B}', '{C}', '{nan}']
		self.assertEqual(r, e)

	def test_creation_categorical_data_enough_bins(self):
		"Test binning of normal categorical data"
		# Create data
		x = ['A'] * 50 + ['B'] * 10 + ['C'] * 20

		# Calculate binning
		with warnings.catch_warnings(record=True) as w:
			bins = create_binning(x=x, nbins=10)

			self.assertEqual(len(w), 1)
			self.assertTrue(issubclass(w[-1].category, UserWarning))
			self.assertTrue(
				'insufficient distinct values' in str(w[-1].message).lower())

		self.assertTrue(isinstance(bins, CategoricalBinning))
		self.assertEqual(len(bins), 3)

		# Test the labels of the bins themselves.
		r = bins.label(x)
		self.assertEqual(r[0], '{A}')
		self.assertEqual(r[50], '{B}')
		self.assertEqual(r[60], '{C}')
		self.assertEqual(r[-1], '{C}')

	def test_creation_categorical_data_less_bins(self):
		"Test binning of normal categorical data"

		x = ['A'] * 50 + ['B'] * 10 + ['C'] * 20

		bins = create_binning(x=x, nbins=2)
		self.assertEqual(len(bins), 2)

		r = bins.label(x)
		self.assertEqual(r[0], '{A}')
		self.assertEqual(r[50], '{B,C}')
		self.assertEqual(r[60], '{B,C}')
		self.assertEqual(r[-1], '{B,C}')

	def test_categorical_with_unseen_data(self):
		"Test categorical binning with unseen data"

		x = ['A'] * 50 + ['B'] * 10 + ['C'] * 20
		y = ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 + ['A,B'] * 10 + ['AB'] * 10

		bins = create_binning(x=x, nbins=3)
		self.assertEqual(len(bins), 3)

		r = bins.label(y)
		self.assertEqual(r[0], '{A}')
		self.assertEqual(r[10], '{B}')
		self.assertEqual(r[20], '{C}')
		# now unseen:
		self.assertEqual(r[30], '{unseen}')
		for rr in r[31:]:
			self.assertEqual(rr, '{unseen}')

		r = bins.label(y, '{iter.integer}: {standard}')
		self.assertEqual(r[0], '0: {A}')
		self.assertEqual(r[-1], '?: {unseen}')

	def test_categorical_order(self):
		"Test that categorical bins requiring no merging are created in alphabetical order"
		x = ['C'] * 50 + ['B'] * 10 + ['A'] * 20
		bins = create_binning(x=x, nbins=3)
		r = bins.label(x, '{iter.integer}: {standard}')
		self.assertEqual(r[0], '2: {C}')
		self.assertEqual(r[50], '1: {B}')
		self.assertEqual(r[60], '0: {A}')
