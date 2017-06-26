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
        self.rand_s3 = np.random.poisson(lam=1, size=1000)
        self.rand_s4 = np.random.poisson(lam=3, size=1000)
        self.rand_s5 = np.random.normal(loc=0, size=1000)
        self.rand_s6 = np.random.normal(loc=0.1, size=1000)
        self.rand_s5[0] = np.nan
        self.rand_s6[0] = np.nan
        self.rand_s6[1] = np.nan

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
        res = es.obrien_fleming(np.linspace(0, 1, 5 + 1)[1:])
        expected_res = [1.17264468e-05, 1.94191300e-03, 1.13964185e-02, 2.84296308e-02, 5.00000000e-02]
        np.testing.assert_almost_equal(res, expected_res)

        # Check float as input
        res = es.obrien_fleming(0.5)
        self.assertAlmostEqual(res, 0.005574596680784305)

    def test_group_sequential(self):
        """
        Check the group sequential function.
        """
        stop, delta, CI, n_x, n_y, mu_x, mu_y = es.group_sequential(self.rand_s1, self.rand_s2)
        self.assertEqual(stop, 1)
        self.assertAlmostEqual(delta, -0.15887364780635896)
        # np.testing.assert_almost_equal(CI.values(), [-0.24461812530841959, -0.07312917030429833], decimal=5)
        np.testing.assert_almost_equal(CI[97.5], -0.07312917030429833, decimal=5)
        np.testing.assert_almost_equal(CI[2.5000000000000022], -0.24461812530841959, decimal=5)
        self.assertEqual(n_x, 1000)
        self.assertEqual(n_y, 1000)
        self.assertAlmostEqual(mu_x, -0.045256707490195384)
        self.assertAlmostEqual(mu_y, 0.11361694031616358)


class BayesFactorTestCases(EarlyStoppingTestCase):
    """
      Test cases for the bayes_factor function in core.early_stopping.
      """

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor(self):
        """
        Check the Bayes factor function.
        """
        stop, delta, CI, n_x, n_y, mu_x, mu_y = es.bayes_factor(self.rand_s1, self.rand_s2, num_iters=2000)
        self.assertEqual(stop, 1)
        self.assertAlmostEqual(delta, -0.15887364780635896)
        self.assertAlmostEqual(CI['lower'], -0.24644586591251214)
        self.assertAlmostEqual(CI['upper'], -0.075698253541557695)
        self.assertEqual(n_x, 1000)
        self.assertEqual(n_y, 1000)
        self.assertAlmostEqual(mu_x, -0.045256707490195384)
        self.assertAlmostEqual(mu_y, 0.11361694031616358)

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor_poisson(self):
        """
        Check the Bayes factor function for Poisson distributions.
        """
        stop, delta, CI, n_x, n_y, mu_x, mu_y = es.bayes_factor(self.rand_s3, self.rand_s4, distribution='poisson',
                                                                num_iters=2000)
        self.assertEqual(stop, 1)
        self.assertAlmostEqual(delta, -1.9589999999999999)
        self.assertAlmostEqual(CI['lower'], -2.0713281392132465)
        self.assertAlmostEqual(CI['upper'], -1.8279692168150592)
        self.assertEqual(n_x, 1000)
        self.assertEqual(n_y, 1000)
        self.assertAlmostEqual(mu_x, 0.96599999999999997)
        self.assertAlmostEqual(mu_y, 2.9249999999999998)

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor_with_nan_input(self):
        """
        Check the Bayes factor function with input that contains nan values.
        """
        stop, _, _, _, _, _, _ = es.bayes_factor(self.rand_s5, self.rand_s6)
        self.assertEqual(stop, 1)


class BayesPrecisionTestCases(EarlyStoppingTestCase):
    """
      Test cases for the bayes_precision function in core.early_stopping.
      """

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_precision(self):
        """
        Check the bayes_precision function.
        """
        stop, delta, CI, n_x, n_y, mu_x, mu_y = es.bayes_precision(self.rand_s1, self.rand_s2, num_iters=2000)
        self.assertEqual(stop, 0)
        self.assertAlmostEqual(delta, -0.15887364780635896)
        self.assertAlmostEqual(CI['lower'], -0.24644586591251214)
        self.assertAlmostEqual(CI['upper'], -0.075698253541557695)
        self.assertEqual(n_x, 1000)
        self.assertEqual(n_y, 1000)
        self.assertAlmostEqual(mu_x, -0.045256707490195384)
        self.assertAlmostEqual(mu_y, 0.11361694031616358)


if __name__ == '__main__':
    unittest.main()
