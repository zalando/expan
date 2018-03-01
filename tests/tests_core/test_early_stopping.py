import unittest

import numpy as np

import expan.core.early_stopping as es
from expan.core.util import find_value_by_key_with_condition


class EarlyStoppingTestCase(unittest.TestCase):
    """ Defines the setUp() and tearDown() functions for the early-stopping test cases."""

    def setUp(self):
        """ Load the needed datasets for all EarlyStoppingTestCases and set the random
        seed so that randomized algorithms show deterministic behaviour.
        """
        np.random.seed(0)

        self.rand_s1 = np.random.normal (loc=0.0, size=1000)
        self.rand_s2 = np.random.normal (loc=0.1, size=1000)
        self.rand_s3 = np.random.poisson(lam=1.0, size=1000)
        self.rand_s4 = np.random.poisson(lam=3.0, size=1000)
        self.rand_s5 = np.random.normal (loc=0.0, size=1000)
        self.rand_s6 = np.random.normal (loc=0.1, size=1000)
        self.rand_s5[0] = np.nan
        self.rand_s6[0] = np.nan
        self.rand_s6[1] = np.nan

    def tearDown(self):
        """Clean up after the test"""
        pass


class GroupSequentialTestCases(EarlyStoppingTestCase):
    """ Test cases for the group sequential functions in core.early_stopping."""

    def test_obrien_fleming(self):
        """ Check the O'Brien-Fleming spending function."""
        # Check array as input
        res_1 = es.obrien_fleming(np.linspace(0, 1, 5 + 1)[1:])
        expected_res = [1.17264468e-05, 1.94191300e-03, 1.13964185e-02, 2.84296308e-02, 5.00000000e-02]
        np.testing.assert_almost_equal(res_1, expected_res)

        # Check float as input
        res_2 = es.obrien_fleming(0.5)
        self.assertAlmostEqual(res_2, 0.005574596680784305)

        # Check int as input
        res_3 = es.obrien_fleming(1)
        self.assertAlmostEqual(res_3, 0.05)

    def test_group_sequential(self):
        """ Check the group sequential function."""
        res = es.group_sequential(self.rand_s1, self.rand_s2)

        self.assertEqual(res.treatment_statistics.sample_size,     1000)
        self.assertEqual(res.control_statistics.sample_size,       1000)
        self.assertAlmostEqual(res.treatment_statistics.mean,     -0.045256707490195384)
        self.assertAlmostEqual(res.control_statistics.mean,        0.11361694031616358)
        self.assertAlmostEqual(res.treatment_statistics.variance,  0.9742344563121542)
        self.assertAlmostEqual(res.control_statistics.variance,    0.9373337542827797)

        self.assertAlmostEqual(res.delta, -0.15887364780635896)
        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')
        np.testing.assert_almost_equal(value025, -0.24461812530841959, decimal=5)
        np.testing.assert_almost_equal(value975, -0.07312917030429833, decimal=5)

        self.assertAlmostEqual(res.p,                 0.0002863669955157941)
        self.assertAlmostEqual(res.statistical_power, 0.9529152504960496)
        self.assertEqual(res.stop,                    True)

    def test_group_sequential_actual_size_larger_than_estimated(self):
        """ Check the group sequential function with wrong input,
        such that the actual data size is already larger than estimated sample size.
        """
        res = es.group_sequential(self.rand_s1, self.rand_s2, estimated_sample_size=100)

        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')
        np.testing.assert_almost_equal (value025, -0.24461812530841959, decimal=5)
        np.testing.assert_almost_equal (value975, -0.07312917030429833, decimal=5)


class BayesFactorTestCases(EarlyStoppingTestCase):
    """ Test cases for the bayes_factor function in core.early_stopping."""

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor(self):
        """ Check the Bayes factor function."""
        res = es.bayes_factor(self.rand_s1, self.rand_s2, num_iters=2000)

        self.assertEqual(res.treatment_statistics.sample_size,     1000)
        self.assertEqual(res.control_statistics.sample_size,       1000)
        self.assertAlmostEqual(res.treatment_statistics.mean,     -0.045256707490195384)
        self.assertAlmostEqual(res.control_statistics.mean,        0.11361694031616358)
        self.assertAlmostEqual(res.treatment_statistics.variance,  0.9742344563121542)
        self.assertAlmostEqual(res.control_statistics.variance,    0.9373337542827797)

        self.assertAlmostEqual(res.delta, -0.15887364780635896)
        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')
        np.testing.assert_almost_equal(value025, -0.24293384641452503, decimal=5)
        np.testing.assert_almost_equal(value975, -0.075064346336461404, decimal=5)
        self.assertEqual(res.p,                 None)
        self.assertEqual(res.statistical_power, None)
        self.assertEqual(res.stop,              True)

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor_poisson(self):
        """ Check the Bayes factor function for Poisson distributions."""
        res = es.bayes_factor(self.rand_s3, self.rand_s4, distribution='poisson', num_iters=2000)

        self.assertEqual(res.treatment_statistics.sample_size,      1000)
        self.assertEqual(res.control_statistics.sample_size,        1000)
        self.assertAlmostEqual(res.treatment_statistics.mean,       0.96599999999999997)
        self.assertAlmostEqual(res.control_statistics.mean,         2.9249999999999998)
        self.assertAlmostEqual(res.treatment_statistics.variance,   0.868844)
        self.assertAlmostEqual(res.control_statistics.variance,     2.901375)

        self.assertAlmostEqual(res.delta, -1.9589999999999999)
        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')
        np.testing.assert_almost_equal(value025, -2.0713281392132465, decimal=5)
        np.testing.assert_almost_equal(value975, -1.8279692168150592, decimal=5)
        self.assertEqual(res.p,                 None)
        self.assertEqual(res.statistical_power, None)
        self.assertEqual(res.stop,              True)

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor_with_nan_input(self):
        """ Check the Bayes factor function with input that contains nan values."""
        res = es.bayes_factor(self.rand_s5, self.rand_s6, num_iters=2000)
        self.assertEqual(res.stop, True)

    def test_variational_inference(self):
        """ Check bayesian sampling using variational bayes."""
        traces, n_x, n_y, mu_x, mu_y = es._bayes_sampling(self.rand_s1, self.rand_s2,
                                                          num_iters=2000, inference="variational")
        self.assertEqual(len(traces), 4)
        self.assertEqual(len(traces['delta']), 1001)
        self.assertEqual(n_x, 1000)
        self.assertEqual(n_y, 1000)


class BayesPrecisionTestCases(EarlyStoppingTestCase):
    """ Test cases for the bayes_precision function in core.early_stopping."""

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_precision(self):
        """ Check the bayes_precision function."""
        res = es.bayes_precision(self.rand_s1, self.rand_s2, num_iters=2000)

        self.assertEqual(res.treatment_statistics.sample_size, 1000)
        self.assertEqual(res.control_statistics.sample_size, 1000)
        self.assertAlmostEqual(res.treatment_statistics.mean, -0.045256707490195384)
        self.assertAlmostEqual(res.control_statistics.mean, 0.11361694031616358)
        self.assertAlmostEqual(res.treatment_statistics.variance, 0.9742344563121542)
        self.assertAlmostEqual(res.control_statistics.variance, 0.9373337542827797)

        self.assertAlmostEqual(res.delta, -0.15887364780635896)
        value025 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 2.5, 'value')
        value975 = find_value_by_key_with_condition(res.confidence_interval, 'percentile', 97.5, 'value')
        np.testing.assert_almost_equal(value025, -0.24293384641452503, decimal=5)
        np.testing.assert_almost_equal(value975, -0.07506434633646140, decimal=5)
        self.assertEqual(res.p,                 None)
        self.assertEqual(res.statistical_power, None)
        self.assertEqual(res.stop,              False)

if __name__ == '__main__':
    unittest.main()
