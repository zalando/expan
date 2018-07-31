import numpy as np


def benjamini_hochberg(false_discovery_rate, original_p_values):
    """ Benjamini-Hochberg procedure.

    :param false_discovery_rate: proportion of significant results that are actually false positives
    :type  false_discovery_rate: float
    :param original_p_values: p values from all the tests
    :type  original_p_values: list[float]

    :return: new critical value (i.e. the corrected alpha)
    :rtype: float
    """
    assert original_p_values, "empty array of p values to analyse"

    p_values_sorted = np.sort(np.asarray(original_p_values))
    number_tests = len(original_p_values)
    significant_ranks = [i for i, val in enumerate(p_values_sorted, 1) if val <= i * false_discovery_rate / number_tests]
    rank = np.max(significant_ranks) if significant_ranks else 1
    return rank * false_discovery_rate / number_tests


def bonferroni(false_positive_rate, original_p_values):
    """ Bonferrnoi correction.

    :param false_positive_rate: alpha value before correction
    :type  false_positive_rate: float
    :param original_p_values: p values from all the tests
    :type  original_p_values: list[float]

    :return: new critical value (i.e. the corrected alpha)
    :rtype: float
    """
    assert original_p_values, "empty array of p values to analyse"

    return false_positive_rate / len(original_p_values)
