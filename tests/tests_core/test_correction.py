import unittest

import numpy as np
from expan.core.results import *
from expan.core.correction import *
from expan.core.statistical_test import *
from expan.core.util import find_value_by_key_with_condition


class CorrectionTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_benjamini_hochberg(self):
        false_discovery_rate = 0.05
        original_p_values = [0.1, 0.03, 0.02, 0.04]

        corrected_alpha = benjamini_hochberg(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.0125)

    def test_benjamini_hochberg_one_p_value(self):
        false_discovery_rate = 0.05
        original_p_values = [0.1]

        corrected_alpha = benjamini_hochberg(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.05)

    def test_bonferroni(self):
        false_discovery_rate = 0.05
        original_p_values = [0.1, 0.03, 0.02, 0.03]

        corrected_alpha = benjamini_hochberg(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.0375)

    def test_bonferroni_one_p_value(self):
        false_discovery_rate = 0.05
        original_p_values = [0.1]

        corrected_alpha = benjamini_hochberg(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.05)
