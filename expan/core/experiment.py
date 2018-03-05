import logging

import numpy as np
import pandas as pd
import copy

import expan.core.early_stopping as es
import expan.core.statistics as statx
from expan.core.statistical_test import *
from expan.core.results import StatisticalTestResult, MultipleTestSuiteResult, CorrectedTestStatistics

logger = logging.getLogger(__name__)


class Experiment(object):
    """ Class which adds the analysis functions to experimental data. """
    def __init__(self, data, metadata):
        """
        Constructor of the experiment object.
        
        :param data: all information of the experiment data. (e.g. KPIs, variant, entity, features, etc)
        :type  data: pd.DataFrame
        :param metadata: additional information about the experiment. (e.g. primary KPI, source, etc)
        :type  metadata: dict
        """
        self.data         = data.copy()
        self.metadata     = metadata.copy()
        self.worker_table = {
            'fixed_horizon': statx.make_delta,
            'group_sequential': es.make_group_sequential,
            'bayes_factor': es.make_bayes_factor,
            'bayes_precision': es.make_bayes_precision
        }


    def __str__(self):
        return 'Experiment "{:s}" with {:d} entities.'.format(self.metadata['experiment'], len(self.data))


    def analyze_statistical_test(self, test, testmethod, **worker_args):
        """ Runs delta analysis on one statistical test and returns statistical results.
        
        :param test: a statistical test to run
        :type  test: StatisticalTest
        :param testmethod: analysis method
        :type  testmethod: str
        :param **worker_args: additional arguments for the analysis method

        :return: statistical result of the test
        :rtype: StatisticalTestResult
        """
        if not isinstance(test, StatisticalTest):
            raise TypeError("Statistical test should be of type StatisticalTest.")

        if 'entity' not in self.data.columns:
            raise KeyError("There is no 'entity' column in the data.")
        if self.data.entity.duplicated().any():
            raise ValueError('Entities in data should be unique.')

        if test.variants.variant_column_name not in self.data.columns:
            raise KeyError("There is no '{}' column in the data.".format(test.variants.variant_column_name))
        if test.variants.treatment_name not in pd.unique(self.data[test.variants.variant_column_name]):
            raise KeyError("There is no treatment with the name '{}' in the data.".format(test.variants.treatment_name))
        if test.variants.control_name not in pd.unique(self.data[test.variants.variant_column_name]):
            raise KeyError("There is no control with the name '{}' in the data.".format(test.variants.control_name))

        for feature in test.features:
            if feature.column_name not in self.data.columns:
                raise KeyError("Feature name '{}' does not exist in the data.".format(feature.column_name))

        if type(test.kpi) is KPI and (test.kpi.name not in self.data.columns):
            raise KeyError("There is no column of name '{}' in the data.".format(test.kpi.name))
        if type(test.kpi) is DerivedKPI:
            if type(test.kpi.numerator) is not str or test.kpi.numerator not in self.data.columns:
                raise KeyError("Numerator '{}' of the derived KPI does not exist in the data.".format(test.kpi.numerator))
            if type(test.kpi.denominator) is not str or test.kpi.denominator not in self.data.columns:
                raise KeyError("Denominator '{}' of the derived KPI does not exist in the data.".format(test.kpi.denominator))
            test.kpi.make_derived_kpi(self.data)

        logger.info("One analysis with kpi '{}', control variant '{}', treatment variant '{}' and features [{}] "
                    "has just started".format(test.kpi, test.variants.control_name,
                                              test.variants.treatment_name,
                                              [(feature.column_name, feature.column_value) for feature in test.features]))

        if testmethod not in self.worker_table:
            raise NotImplementedError("Test method '{}' is not implemented.".format(testmethod))
        worker = self.worker_table[testmethod](**worker_args)

        # create test result object with empty result first
        test_result = StatisticalTestResult(test, None)

        data_for_analysis = self.data
        # apply feature filter to data
        for feature in test.features:
            data_for_analysis = feature.apply_to_data(data_for_analysis)

        if not self._is_valid_for_analysis(data_for_analysis, test):
            logger.warning("Data is not valid for the analysis!")
            return test_result

        # get control and treatment values for the kpi
        control          = test.variants.get_variant(data_for_analysis, test.variants.control_name)[test.kpi.name]
        control_weight   = self._get_weights(data_for_analysis, test, test.variants.control_name)
        control_data     = control * control_weight

        treatment        = test.variants.get_variant(data_for_analysis, test.variants.treatment_name)[test.kpi.name]
        treatment_weight = self._get_weights(data_for_analysis, test, test.variants.treatment_name)
        treatment_data   = treatment * treatment_weight

        # run the test method
        test_statistics = worker(x=treatment_data, y=control_data)
        test_result.result = test_statistics
        return test_result


    def analyze_statistical_test_suite(self, test_suite, testmethod='fixed_horizon', **worker_args):
        """ Runs delta analysis on a set of tests and returns statistical results for each statistical test in the suite.
        
        :param test_suite: a suite of statistical test to run
        :type  test_suite: StatisticalTestSuite
        :param testmethod: analysis method to perform. 
                           It can be 'fixed_horizon', 'group_sequential', 'bayes_factor' or 'bayes_precision'.
        :type  testmethod: str
        :param **worker_args: additional arguments for the analysis method (see signatures of corresponding methods)

        :return: statistical result of the test suite
        :rtype: MultipleTestSuiteResult
        """
        if not isinstance(test_suite, StatisticalTestSuite):
            raise TypeError("Test suite should be of type StatisticalTestSuite.")

        if testmethod not in ['fixed_horizon', 'group_sequential']:
            test_suite.correction_method = CorrectionMethod.NONE
        requires_correction = test_suite.correction_method is not CorrectionMethod.NONE

        # look up table for correction method
        correction_table = {
            CorrectionMethod.BONFERRONI: bonferroni,
            CorrectionMethod.BH: benjamini_hochberg
        }

        # test_suite_result hold statistical results from all statistical tests
        test_suite_result = MultipleTestSuiteResult([], test_suite.correction_method)
        for test in test_suite.tests:
            original_analysis = self.analyze_statistical_test(test, testmethod, **worker_args)
            test_suite_result.results.append(original_analysis)

        # if correction is needed, get p values, do correction on alpha, and run the same analysis for new alpha
        if requires_correction:
            original_alpha    = worker_args.get('alpha', 0.05)
            original_p_values = [item.result.p for item in test_suite_result.results]
            corrected_alpha   = correction_table[test_suite.correction_method](original_alpha, original_p_values)
            new_worker_args   = copy.deepcopy(worker_args)
            new_worker_args['alpha'] = corrected_alpha

            for test_index, test in enumerate(test_suite.tests):
                original_analysis        = test_suite_result.results[test_index]
                corrected_analysis       = self.analyze_statistical_test(test, testmethod, **new_worker_args)
                combined_result          = CorrectedTestStatistics(original_analysis.result, corrected_analysis.result)
                original_analysis.result = combined_result

        return test_suite_result


    def outlier_filter(self, kpis, percentile=99.0, threshold_type='upper'):
        """ Method that filters out entities whose KPIs exceed the value at a given percentile.
        If any of the KPIs exceeds its threshold the entity is filtered out.
        
        :param kpis: list of KPI names
        :type  kpis: list[str]
        :param percentile: percentile considered as filtering threshold
        :type  percentile: float
        :param threshold_type: type of threshold used ('lower' or 'upper')
        :type  threshold_type: str

        :return: No return value. Will filter out outliers in self.data in place.
        """
        # check if provided KPIs are present in the data
        for kpi in kpis:
            if kpi not in self.data.columns:
                raise KeyError(kpi + ' identifier not present in dataframe columns!')
        # check if provided percentile is valid
        if 0.0 < percentile <= 100.0 is False:
            raise ValueError("Percentile value needs to be between 0.0 and 100.0!")
        # check if provided filtering kind is valid
        if threshold_type not in ['upper', 'lower']:
            raise ValueError("Threshold type needs to be either 'upper' or 'lower'!")

        # run quantile filtering
        flags = self._quantile_filtering(kpis=kpis, percentile=percentile, threshold_type=threshold_type)
        # log which columns were filtered and how many entities were filtered out
        self.metadata['filtered_columns'] = kpis
        self.metadata['filtered_entities_number'] = len(flags[flags == True])
        self.metadata['filtered_threshold_kind'] = threshold_type
        # throw warning if too many entities have been filtered out
        if (len(flags[flags == True]) / float(len(self.data))) > 0.02:
            logger.warning('More than 2% of entities have been filtered out, consider adjusting the percentile value.')
        self.data = self.data[flags == False]


    # ----- below are helper methods ----- #
    def _is_valid_for_analysis(self, data, test):
        """ Check whether the quality of data is good enough to perform analysis. Invalid cases can be:
        1. there is no data
        2. the data does not contain all the variants to perform analysis
        
        :type data: DataFrame
        :type test: StatisticalTest
        :rtype: bool 
        """
        if len(data[data[test.variants.variant_column_name] == test.variants.control_name]) <= 1:
            logger.warning("Control group only contains 1 or 0 entities.")
            return False
        if len(data[data[test.variants.variant_column_name] == test.variants.treatment_name]) <= 1:
            logger.warning("Treatment group only contains 1 or 0 entities.")
        return True


    def _get_weights(self, data, test, variant_name):
        """ Perform the re-weighting trick. 
        See http://expan.readthedocs.io/en/latest/glossary.html#per-entity-ratio-vs-ratio-of-totals
        
        :type data: pd.DataFrame
        :type test: StatisticalTest
        :type variant_name: str
        :rtype: pd.DataFrame
        """
        if type(test.kpi) is not DerivedKPI:
            return 1.0

        x = test.variants.get_variant(data, variant_name)
        number_of_zeros_and_nans     = sum(x == 0) + np.isnan(x).sum()
        number_of_non_zeros_and_nans = len(x) - number_of_zeros_and_nans
        return number_of_non_zeros_and_nans/np.nansum(x) * x


    def _quantile_filtering(self, kpis, percentile, threshold_type):
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
        method_table = {'upper': lambda x: x > threshold, 'lower': lambda x: x <= threshold}
        flags = pd.Series(data=[False]*len(self.data))
        for column in self.data[kpis].columns:
            threshold = np.percentile(self.data[column], percentile)
            flags = flags | self.data[column].apply(method_table[threshold_type])
        return flags
