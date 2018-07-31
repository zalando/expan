import unittest
from expan.core.correction import benjamini_hochberg, bonferroni


class CorrectionTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_benjamini_hochberg(self):
        false_discovery_rate = 0.05

        original_p_values = [0.02, 0.03, 0.002, 0.001]
        corrected_alpha = benjamini_hochberg(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.05)

        original_p_values = [0.01, 0.2, 0.1, 0.02]
        corrected_alpha = benjamini_hochberg(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.025)

    def test_benjamini_hochberg_one_p_value(self):
        false_discovery_rate = 0.05
        original_p_values = [0.1]

        corrected_alpha = benjamini_hochberg(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.05)

    def test_benjamini_hochberg_empty(self):
        with self.assertRaises(AssertionError):
            _ = benjamini_hochberg(0.5, [])

    def test_bonferroni(self):
        false_discovery_rate = 0.05
        original_p_values = [0.1, 0.03, 0.02, 0.03]

        corrected_alpha = bonferroni(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.0125)

    def test_bonferroni_one_p_value(self):
        false_discovery_rate = 0.05
        original_p_values = [0.1]

        corrected_alpha = bonferroni(false_discovery_rate, original_p_values)
        self.assertAlmostEqual(corrected_alpha, 0.05)

    def test_bonferroni_empty(self):
        with self.assertRaises(AssertionError):
            _ = bonferroni(0.5, [])
