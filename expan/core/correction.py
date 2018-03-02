import logging
from expan.core.results import MultipleTestSuiteResult, CorrectedTestStatistics

logger = logging.getLogger(__name__)


def add_corrected_test_statistics(test_suite_results):
    """
    Takes statistical suite results and add to the results the corrected test statistics 
    with adjusted p values and confidence intervals.
    
    :param test_suite_results: instance of type MultipleTestSuiteResult
    :type  test_suite_results: MultipleTestSuiteResult
    
    :return: corrected statistical test suite results
    :rtype:  MultipleTestSuiteResult
    """
    if not isinstance(test_suite_results, MultipleTestSuiteResult):
        raise RuntimeError("Please provide test suite results of type MultipleTestSuiteResult")

    correction_methods = {
        'bf': bf_correction,
        'bh': bh_correction
    }

    # Suite results
    suite_results_statistics = test_suite_results.statistical_test_results

    # Take all p values from each statistical result
    p_values = [test_suite_result.result.p for test_suite_result in suite_results_statistics]
    confidence_intervals = [test_suite_result.result.confidence_interval for test_suite_result in suite_results_statistics]

    # Correct p-values with the correction method
    corrected_p_values, corrected_confidence_intervals = \
        correction_methods[test_suite_results.correction_method](p_values, confidence_intervals)

    # List preserves the order. For each statistical result create CorrectedTestStatistics with original_test_statistics
    # and corrected_test_statistics. original_test_statistics is a copy of the test result, corrected_test_statistics
    # is a copy of that plus updated p values and confidence intervals.
    for i in range(len(corrected_p_values)):
        corrected_test_results = CorrectedTestStatistics(suite_results_statistics[i].result,
                                                         suite_results_statistics[i].result)
        corrected_test_results.corrected_test_statistics.p = corrected_p_values[i]
        corrected_test_results.corrected_test_statistics.confidence_interval = corrected_confidence_intervals[i]
        test_suite_results.statistical_test_results[i].result = corrected_test_results

    return test_suite_results


def bf_correction(p_values, confidence_intervals):
    """ Corrects p values and confidence intervals with Bonferroni multiple correction adjustment.

    :param p_values: p values for correction
    :rtype p_values: list
    :param confidence_intervals: confidence intervals
    :rtype confidence_intervals: list[dict]

    :return: adjusted p values and confidence intervals
    :rtype:  list
    """
    corrected_p_values = [p_value*len(p_values) for p_value in p_values]

    for ci in confidence_intervals:
        ci['percentile'] = float(ci['percentile']) / len(p_values) if ci['percentile'] < 50.0 \
            else 100 - (100 - float(ci['percentile'])) / len(p_values) if ci['percentile'] > 50.0 else ci['percentile']

    return corrected_p_values, confidence_intervals


def bh_correction(p_values):
    """ Corrects p values and confidence intervals with Benjamini-Hochberg multiple correction adjustment.
    
    :param p_values: p values for correction
    :rtype p_values: list

    :return: adjusted p values and confidence intervals
    :rtype:  list
    """
    pass
