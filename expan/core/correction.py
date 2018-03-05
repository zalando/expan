import logging
import numpy as np

logger = logging.getLogger(__name__)


def benjamini_hochberg(false_discovery_rate, original_p_values):
    """ Benjamini–Hochberg procedure.
    
    :param false_discovery_rate: proportion of significant results that are actually false positives
    :type  false_discovery_rate: float
    :param original_p_values: p values from all the tests
    :type  original_p_values: list[float]
    
    :return: new critical value (i.e. the corrected alpha)
    :rtype: float
    """
    pvals = np.asarray(original_p_values)
    pvals_sorted = np.sort(pvals)
    numtests = len(original_p_values)
    if len([i for i, val in enumerate(pvals_sorted, 1) if val <= i * false_discovery_rate / numtests]) != 0:
        R = np.max([i for i, val in enumerate(pvals_sorted, 1) if val <= i * false_discovery_rate / numtests])
    else:
        R = 1
    return R * false_discovery_rate / numtests


def bonferroni(false_positive_rate, original_p_values):
    """ Bonferrnoi correction.
    
    :param false_positive_rate: alpha value before correction
    :type  false_positive_rate: float
    :param original_p_values: p values from all the tests
    :type  original_p_values: list[float]
    
    :return: new critical value (i.e. the corrected alpha)
    :rtype: float
    """
    return false_positive_rate / len(original_p_values)
