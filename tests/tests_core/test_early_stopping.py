import unittest
import numpy as np
import expan.core.early_stopping as es


class EarlyStoppingTestCase(unittest.TestCase):
	"""
  	Defines the setUp() and tearDown() functions for the early-stopping test cases.
  	"""

	def setUp(self):
		"""
    	Load the needed datasets for all EarlyStoppingTestCases and set the random
    	seed so that randomized algorithms show deterministic behaviour.
    	"""
		np.random.seed(0)

		self.rand_s1 = np.random.normal(loc=0, size=1000)
		self.rand_s2 = np.random.normal(loc=0.1, size=1000)

	def tearDown(self):
		"""
    	Clean up after the test
    	"""
		# TODO: find out if we have to remove data manually
		pass


class GroupSequentialTestCases(EarlyStoppingTestCase):
	"""
  	Test cases for the group sequential functions in core.early_stopping.
  	"""

	def test_obrien_fleming(self):
		"""
    	Check the O'Brien-Fleming spending function.
    	"""
		# Check array as input
		res = es.obrien_fleming(np.linspace(0,1,5+1)[1:])
		np.testing.assert_almost_equal(res, [1.17264468e-05,1.94191300e-03,1.13964185e-02,2.84296308e-02,5.00000000e-02])
		# Check float as input
		res = es.obrien_fleming(0.5)
		self.assertAlmostEqual(res, 0.005574596680784305)

	def test_group_sequential(self):
		"""
    	Check the group sequential function.
    	"""
		stop,delta,CI,n_x,n_y,mu_x,mu_y = es.group_sequential(self.rand_s1, self.rand_s2)
		self.assertEqual(stop, True)
		self.assertAlmostEqual(delta, -0.15887364780635896)
		self.assertEqual(len(CI), 0)
		self.assertEqual(n_x, 1000)
		self.assertEqual(n_y, 1000)
		self.assertAlmostEqual(mu_x, -0.045256707490195384)
		self.assertAlmostEqual(mu_y, 0.11361694031616358)


class BayesFactorTestCases(EarlyStoppingTestCase):
	"""
  	Test cases for the bayes_factor function in core.early_stopping.
  	"""

	def test_bayes_factor(self):
		"""
    	Check the Bayes factor function.
    	"""
		stop,delta,CI,n_x,n_y,mu_x,mu_y = es.bayes_factor(self.rand_s1, self.rand_s2)
		self.assertEqual(stop, True)
		self.assertAlmostEqual(delta, -0.15887364780635896)
		self.assertAlmostEqual(CI['lower'], -0.24414725578976518)
		self.assertAlmostEqual(CI['upper'], -0.072120687308212819)
		self.assertEqual(n_x, 1000)
		self.assertEqual(n_y, 1000)
		self.assertAlmostEqual(mu_x, -0.045256707490195384)
		self.assertAlmostEqual(mu_y, 0.11361694031616358)


if __name__ == '__main__':
	unittest.main()
