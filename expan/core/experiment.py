import logging
import warnings

import numpy as np
import copy

import expan.core.early_stopping as es
import expan.core.statistics as statx
import expan.core.correction as correction
from expan.core.statistical_test import *
from expan.core.results import StatisticalTestResult, MultipleTestSuiteResult, CombinedTestStatistics

warnings.simplefilter('always', UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class Experiment(object):
    """ Class which adds the analysis functions to experimental data. """
    def __init__(self, metadata):
        """
        Constructor of the experiment object.
        
        :param metadata: additional information about the experiment. (e.g. primary KPI, source, etc)
        :type  metadata: dict
        """
        self.metadata     = metadata
        self.worker_table = {
            'fixed_horizon': statx.make_delta,
            'group_sequential': es.make_group_sequential,
            'bayes_factor': es.make_bayes_factor,
            'bayes_precision': es.make_bayes_precision
        }

    def __str__(self):
        return 'Performing "{:s}" experiment.'.format(self.metadata['experiment'])


    def analyze_statistical_test(self, test, test_method='fixed_horizon', include_data=False, **worker_args):
        """ Runs delta analysis on one statistical test and returns statistical results.
        
        :param test: a statistical test to run
        :type  test: StatisticalTest
        :param test_method: analysis method to perform. 
                           It can be 'fixed_horizon', 'group_sequential', 'bayes_factor' or 'bayes_precision'.
        :type  test_method: str
        :param include_data: True if test results should include data, False - if no data should be included
        :type  include_data: bool
        :param worker_args: additional arguments for the analysis method

        :return: statistical result of the test
        :rtype: StatisticalTestResult
        """
        if not isinstance(test, StatisticalTest):
            raise TypeError("Statistical test should be of type StatisticalTest.")

        if 'entity' not in test.data.columns:
            raise KeyError("There is no 'entity' column in the data.")
        if test.variants.variant_column_name not in test.data.columns:
            raise KeyError("There is no '{}' column in the data.".format(test.variants.variant_column_name))

        for feature in test.features:
            if feature.column_name not in test.data.columns:
                raise KeyError("Feature name '{}' does not exist in the data.".format(feature.column_name))

        if type(test.kpi) is KPI and (test.kpi.name not in test.data.columns):
            raise KeyError("There is no column of name '{}' in the data.".format(test.kpi.name))
        if type(test.kpi) is DerivedKPI:
            if type(test.kpi.numerator) is not str or test.kpi.numerator not in test.data.columns:
                raise KeyError("Numerator '{}' of the derived KPI does not exist in the data.".format(test.kpi.numerator))
            if type(test.kpi.denominator) is not str or test.kpi.denominator not in test.data.columns:
                raise KeyError("Denominator '{}' of the derived KPI does not exist in the data.".format(test.kpi.denominator))
            test.kpi.make_derived_kpi(test.data)

        if test_method not in self.worker_table:
            raise NotImplementedError("Test method '{}' is not implemented.".format(test_method))
        worker = self.worker_table[test_method](**worker_args)

        # create test result object with empty result first
        test_result = StatisticalTestResult(test, None)

        data_for_analysis = test.data
        for feature in test.features:
            data_for_analysis = feature.apply_to_data(data_for_analysis)

        if not self._is_valid_for_analysis(data_for_analysis, test):
            logger.warning("Data is not valid for the analysis!")
            return test_result
        if data_for_analysis.entity.duplicated().any():
            raise ValueError('Entities in data should be unique.')

        # get control and treatment values for the kpi
        control          = test.variants.get_variant(data_for_analysis, test.variants.control_name)[test.kpi.name]
        control_denominators   = self._get_denominators(data_for_analysis, test, test.variants.control_name)
        control_numerators   = control * control_denominators

        treatment        = test.variants.get_variant(data_for_analysis, test.variants.treatment_name)[test.kpi.name]
        treatment_denominators = self._get_denominators(data_for_analysis, test, test.variants.treatment_name)
        treatment_numerators   = treatment * treatment_denominators

        # run the test method
        test_statistics = worker(x=treatment_numerators, y=control_numerators, x_denominators = treatment_denominators, y_denominators = control_denominators)
        test_result.result = test_statistics

        # remove data from the result test metadata
        if not include_data:
            del test.data
        return test_result


    def analyze_statistical_test_suite(self, test_suite, test_method='fixed_horizon', **worker_args):
        """ Runs delta analysis on a set of tests and returns statistical results for each statistical test in the suite.
        
        :param test_suite: a suite of statistical test to run
        :type  test_suite: StatisticalTestSuite
        :param test_method: analysis method to perform. 
                           It can be 'fixed_horizon', 'group_sequential', 'bayes_factor' or 'bayes_precision'.
        :type  test_method: str
        :param worker_args: additional arguments for the analysis method (see signatures of corresponding methods)

        :return: statistical result of the test suite
        :rtype: MultipleTestSuiteResult
        """
        if not isinstance(test_suite, StatisticalTestSuite):
            raise TypeError("Test suite should be of type StatisticalTestSuite.")

        if test_method not in ['fixed_horizon', 'group_sequential']:
            test_suite.correction_method = CorrectionMethod.NONE
        requires_correction = test_suite.correction_method is not CorrectionMethod.NONE

        # look up table for correction method
        correction_table = {
            CorrectionMethod.BONFERRONI: correction.bonferroni,
            CorrectionMethod.BH: correction.benjamini_hochberg
        }

        logger.info("Statistical test suite analysis with {} tests, testmethod {}, correction method {} "
                    "has just started".format(len(test_suite.tests), test_method, test_suite.correction_method))

        # test_suite_result hold statistical results from all statistical tests
        test_suite_result = MultipleTestSuiteResult([], test_suite.correction_method)
        for test in test_suite.tests:
            original_analysis = self.analyze_statistical_test(test, test_method, True, **worker_args)
            # If the statistical power is -1 *or* if the results are None, then we don't include the analysis
            if original_analysis.result is not None and original_analysis.result.statistical_power != -1:
                combined_result = CombinedTestStatistics(original_analysis.result, original_analysis.result)
                original_analysis.result = combined_result
                test_suite_result.results.append(original_analysis)
            else:
                logger.warning("Analysis results are excluded from the result file because they contain Null values.")

        # if correction is needed, get p values, do correction on alpha, and run the same analysis for new alpha
        if requires_correction:
            original_alpha    = worker_args.get('alpha', 0.05)
            original_p_values = [item.result.original_test_statistics.p for item in test_suite_result.results
                                 if item.result.original_test_statistics is not None]
            corrected_alpha   = correction_table[test_suite.correction_method](original_alpha, original_p_values)
            new_worker_args   = copy.deepcopy(worker_args)
            new_worker_args['alpha'] = corrected_alpha

            for test_index, test_item in enumerate(test_suite_result.results):
                if test_item.result.original_test_statistics:  # result can be None if not enough entities
                    original_analysis = test_suite_result.results[test_index]
                    corrected_analysis = self.analyze_statistical_test(test_item.test, test_method,
                                                                       True, **new_worker_args)
                    combined_result = CombinedTestStatistics(original_analysis.result.original_test_statistics,
                                                             corrected_analysis.result)
                    original_analysis.result = combined_result

        logger.info("Statistical test suite analysis with {} tests, testmethod {}, correction method {} "
                    "has finished".format(len(test_suite.tests), test_method, test_suite.correction_method))

        # remove data from the results
        for test_result in test_suite_result.results:
            del test_result.test.data
        return test_suite_result


    def outlier_filter(self, data, kpis, percentile=99.0, threshold_type='upper'):
        """ Method that filters out entities whose KPIs exceed the value at a given percentile.
        If any of the KPIs exceeds its threshold the entity is filtered out. 
        If kpis contains derived kpi, this method will first create these columns,
        and then perform outlier filtering on all given kpis.
        
        :param kpis: list of KPI instances
        :type  kpis: list[KPI]
        :param percentile: percentile considered as filtering threshold
        :type  percentile: float
        :param threshold_type: type of threshold used ('lower' or 'upper')
        :type  threshold_type: str

        :return: Will return data with filtered outliers.
        """
        # check if provided KPIs are present in the data
        for kpi in kpis:
            if type(kpi) is KPI and (kpi.name not in data.columns):
                raise KeyError(kpi.name + ' identifier not present in dataframe columns in outlier filtering!')
            if type(kpi) is DerivedKPI:
                if type(kpi.numerator) is not str or kpi.numerator not in data.columns:
                    raise KeyError(
                        "Numerator '{}' of the derived KPI does not exist in the data in outlier filtering.".format(kpi.numerator))
                if type(kpi.denominator) is not str or kpi.denominator not in data.columns:
                    raise KeyError(
                        "Denominator '{}' of the derived KPI does not exist in the data in outlier filtering.".format(kpi.denominator))
                kpi.make_derived_kpi(data)

        # check if provided percentile is valid
        if 0.0 < percentile <= 100.0 is False:
            raise ValueError("Percentile value needs to be between 0.0 and 100.0!")
        # check if provided filtering kind is valid
        if threshold_type not in ['upper', 'lower']:
            raise ValueError("Threshold type needs to be either 'upper' or 'lower'!")

        # run quantile filtering
        flags = self._quantile_filtering(data=data,
                                         kpis=[kpi.name for kpi in kpis],
                                         percentile=percentile,
                                         threshold_type=threshold_type)
        # log which columns were filtered and how many entities were filtered out
        self.metadata['filtered_columns'] = [kpi.name for kpi in kpis]
        self.metadata['filtered_entities_number'] = len(flags[flags == True])

        filtered = [item[1] for item in list(zip(flags, data['variant'])) if item[0] == True]
        self.metadata['filtered_entities_per_variant'] = dict((val, filtered.count(val)) for val in set(filtered))

        self.metadata['filtered_threshold_kind'] = threshold_type
        # throw warning if too many entities have been filtered out
        if (len(flags[flags == True]) / float(len(data))) > 0.02:
            warnings.warn('More than 2% of entities have been filtered out, consider adjusting the percentile value.')
            logger.warning('More than 2% of entities have been filtered out, consider adjusting the percentile value.')
        return data[flags == False]


    # ----- below are helper methods ----- #
    def _is_valid_for_analysis(self, data, test):
        """ Check whether the quality of data is good enough to perform analysis. Invalid cases can be:
        1. there is no data
        2. the data does not contain all the variants to perform analysis
        
        :param data: data frame for which a check for validity will be made
        :type  data: DataFrame
        :param test: a statistical test for control name and treatment name
        :type  test: StatisticalTest
        
        :return True if data is valid for analysis and False if not
        :rtype: bool 
        """
        count_controls   = sum(data[test.variants.variant_column_name] == test.variants.control_name)
        count_treatments = sum(data[test.variants.variant_column_name] == test.variants.treatment_name)
        if count_controls <= 1:
            logger.warning("Control group only contains {} entities.".format(count_controls))
            return False
        if count_treatments <= 1:
            logger.warning("Treatment group only contains {} entities.".format(count_treatments))
            return False
        return True


    def _get_denominators(self, data, test, variant_name):
        if type(test.kpi) is not DerivedKPI:
            return 1.0

        x = test.variants.get_variant(data, variant_name)[test.kpi.denominator]
        return x


    def _quantile_filtering(self, data, kpis, percentile, threshold_type):
        # initialize 'flags' to a boolean Series (false) with the correct index.
        # By using the correct index, we remove the annoying warnings.
        flags = data.index.to_series() != data.index.to_series()

        from sys import float_info

        """ Make the filtering based on the given quantile level. 
        Filtering is performed for each kpi independently.
        
        :param kpis: the kpis to perform filtering
        :type  kpis: list[str]
        :param percentile: percentile considered as filtering threshold
        :type  percentile: float
        :param threshold_type: type of threshold used ('lower' or 'upper')
        :type  threshold_type: str

        :return: boolean values indicating whether the row should be filtered
        :rtype: pd.Series
        """
        method_table   = {'upper': lambda x: x > threshold, 'lower': lambda x: x <= threshold}
        na_replacement = {'upper': -float_info.max,         'lower':  float_info.max}

        for column in data[kpis].columns:
            threshold = np.percentile(data[column].fillna(na_replacement[threshold_type]), percentile)
            flags = flags | data[column].apply(method_table[threshold_type])
        return flags


    def chi_square_test_result_and_statistics(self, variant_column, weights, min_counts=5, alpha=0.05):
        """ Tests the consistency of variant split with the hypothesized distribution.
        
        :param variant_column: variant column from the input data frame
        :param weights: dict with variant names as keys, weights as values
                        ({<variant_name>:<weight>, ...}
        :param min_counts: minimum number of observed and expected frequencies (should be at least 5), see 
                            http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.chisquare.html
        :param alpha: significance level, 0.05 by default
        :return: True(if split is consistent with the given split) or
                 False(if split is not consistent with the given split)
        :rtype:  Boolean, float, float
        """
        if not hasattr(variant_column, '__len__'):
            raise ValueError("Variant split check was cancelled since input variant column is empty or doesn't exist.")
        if not hasattr(weights, '__len__'):
            raise ValueError("Variant split check was cancelled since input weights are empty or doesn't exist.")
        if len(weights) <= 1 or len(variant_column) <= 1:
            raise ValueError("Variant split check was cancelled since input weights or the number if categories "
                             "is less than 2.")

        # Count number of observations per each variant
        variant_column = pd.Series(variant_column).dropna(axis=0)
        observed_freqs = variant_column.value_counts()

        # Ensure at least a frequency of min_counts at every location in observed_counts.
        # It's recommended to not conduct test if frequencies in each category is less than min_counts
        if len(observed_freqs[observed_freqs < min_counts]) >= 1:
            raise ValueError("Chi-square test is not valid for small expected or observed frequencies.")

        # If there are less than 2 categories left after dropping counts less than 5 we can't conduct the test.
        if len(observed_freqs) < 2:
            raise ValueError("If the number of categories is less than 2 Chi-square test is not applicable.")

        # Calculate expected counts given corresponding weights,
        # weights are filtered out of categories which were dropped before.
        total_count = observed_freqs.sum()
        weights = {k: v for (k, v) in weights.items() if k in observed_freqs.index.values}
        expected_freqs = pd.Series(weights)
        expected_freqs *= total_count

        # Compute chi-square and p-value statistics
        chi_square_val, p_val = statx.chi_square(observed_freqs.sort_index(), expected_freqs.sort_index())

        return p_val >= alpha, p_val, chi_square_val
