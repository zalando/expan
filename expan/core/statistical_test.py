from enum import Enum
import logging

import pandas as pd
from expan.core.util import JsonSerializable

from copy import deepcopy
import numpy as np

from expan.core.correction import CorrectionMethod

import expan.core.early_stopping as es
import expan.core.statistics as statx

from expan.core.results import StatisticalTestResult

logger = logging.getLogger(__name__)


class StatisticalTest(JsonSerializable):
    """ This class describes what has to be tested against what and represent a unit of statistical testing.
    
    :param data: data for statistical test
    :type  data: DataFrame
    :param kpi: the kpi to perform on
    :type  kpi: KPI or its subclass
    :param features: list of features used for subgroups
    :type  features: list[FeatureFilter]
    :param variants: variant column name and their values
    :type  variants: Variants
    """
    def __init__(self, data, kpi, features, variants):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Please, provide data for the statistical test in form of a data frame.")
        if not isinstance(features, list):
            raise TypeError("Features should be a list.")
        if not all(isinstance(n, FeatureFilter) for n in features):
            raise TypeError("Some features are not of the type FeatureFilter.")
        self.data     = data
        self.kpi      = kpi
        self.features = features
        self.variants = variants
        self.data_for_analysis = None
        self.worker_table = {
            'fixed_horizon': statx.make_delta,
            'group_sequential': es.make_group_sequential,
            'bayes_factor': es.make_bayes_factor,
            'bayes_precision': es.make_bayes_precision
        }

    def get_data_for_analysis(self):
        """ Filter out rows from original dataframe which don't have requested features.
    
        :return filtered dataframe applicable for this StatisticalTest
        :rtype pandas.DataFrame
        """
        if self.data_for_analysis is None:
            self.data_for_analysis = self.data
            for feature in self.features:
                self.data_for_analysis = feature.apply_to_data(self.data_for_analysis)
        return self.data_for_analysis
    
    def is_valid_for_analysis(self):
        """ Check whether the quality of data is good enough to perform analysis. Invalid cases can be:
        1. there is no data
        2. the data does not contain all the variants to perform analysis
        
        :return True if data is valid for analysis and False if not
        :rtype: bool 
        """
        data = self.get_data_for_analysis()
        count_controls   = sum(data[self.variants.variant_column_name] == self.variants.control_name)
        count_treatments = sum(data[self.variants.variant_column_name] == self.variants.treatment_name)
        if count_controls <= 1:
            logger.warning("Control group only contains {} entities.".format(count_controls))
            return False
        if count_treatments <= 1:
            logger.warning("Treatment group only contains {} entities.".format(count_treatments))
            return False
        return True

    def split_data_on_control_and_treatment(self):
        data = self.get_data_for_analysis()
        controldf   = self.variants.get_control_variant(data)
        treatmentdf = self.variants.get_treatment_variant(data)

        (control_numerator, control_denominator) = self.kpi.get_numerator_and_denominator(controldf)
        (treatment_numerator, treatment_denominator) = self.kpi.get_numerator_and_denominator(treatmentdf)

        return [(control_numerator, control_denominator), (treatment_numerator, treatment_denominator)]

    def is_applicable_to_data(self):
        if 'entity' not in self.data.columns:
            raise KeyError("There is no 'entity' column in the data.")
        self.variants.exists_in_data(self.data)
        [f.exists_in_data(self.data) for f in self.features]
        self.kpi.exists_in_data(self.data)

    def do_analysis(self, test_method, include_data, **worker_args):
        self.is_applicable_to_data() #will throw otherwise
        
        self.kpi.make_derived_kpi(self.data)
        logger.info("One analysis with kpi '{}', control variant '{}', treatment variant '{}' and features [{}] "
                    "has just started".format(self.kpi, self.variants.control_name,
                                              self.variants.treatment_name,
                                              [(feature.column_name, feature.column_value) for feature in self.features]))
        
        if test_method not in self.worker_table:
            raise NotImplementedError("Test method '{}' is not implemented.".format(test_method))
        worker = self.worker_table[test_method](**worker_args)
        
        # create test result object with empty result first
        test_result = StatisticalTestResult(self, None)

        data_for_analysis = self.get_data_for_analysis()

        if not self.is_valid_for_analysis():
            # Note that this does not check that there are enough
            # non-NaN and non=Inf datapoints. See below for a check
            # of that:
            logger.warning("Data is not valid for the analysis!")
            return test_result
        if data_for_analysis.entity.duplicated().any():
            raise ValueError('Entities in data should be unique.')

        
        # get control and treatment values for the kpi
        [(control_numerators, control_denominators), (treatment_numerators, treatment_denominators)] = self.split_data_on_control_and_treatment()
        logger.info("Control group size: {}".format(control_numerators.shape[0]))

        logger.info("Treatment group size: {}".format(treatment_numerators.shape[0]))

        number_of_finite_controls   = np.sum(np.isfinite( control_numerators   / control_denominators   ))
        number_of_finite_treatments = np.sum(np.isfinite( treatment_numerators / treatment_denominators ))

        if number_of_finite_controls < 2 or number_of_finite_treatments < 2: return test_result

        # run the test method
        test_statistics = worker(x=treatment_numerators, y=control_numerators, x_denominators = treatment_denominators, y_denominators = control_denominators)
        test_result.result = test_statistics

        # remove data from the result test metadata
        if not include_data:
            del self.data
        return test_result



    def __deepcopy__(self, forward_me_to_recursive_deepcopy):
        # This provides a custom 'deepcopy' for this type. See
        # https://docs.python.org/3/library/copy.html#copy.deepcopy for more.

        # We should not 'deepcopy' the pandas Dataframe. The dataframe is
        # essentially immutable for the purposes of this.
        # We'll 'deepcopy' everything else though.

        # TODO: Maybe nothing should be 'deepcopy'-ed here?

        return StatisticalTest(
                self.data,  # deliberately not copied
                deepcopy(self.kpi, forward_me_to_recursive_deepcopy),
                deepcopy(self.features, forward_me_to_recursive_deepcopy),
                deepcopy(self.variants, forward_me_to_recursive_deepcopy),
                )


class KPI(JsonSerializable):
    """ This class represents a basic kpi.
    :param name: name of the kpi
    :type  name: str
    """
    def __init__(self, name):
        self.name = name

    def get_numerator_and_denominator(self, data):
        """Select two columns from a dataset.
        First corresponds to a numerator of a kpi.
        Second corresponds to a denominator of a kpi. In case of a direct KPI - np.float64(1.0)
        
        :param data: dataframe to extract kpi columns from
        :type  data: pandas.DataFrame

        :return: (numerator, denominator) tuple
        :rtype: (pandas.DataFrame, pandas.DataFrame)
        """
        numerator = data[self.name]
        denominator = np.float64(1.0)
        return (numerator, denominator)

    def exists_in_data(self, data):
        if self.name not in data.columns:
            raise KeyError("There is no column of name '{}' in the data.".format(self.name))
        return True
    
    def make_derived_kpi(self, data):
        """ Do nothing for direct KPI """
        pass


class DerivedKPI(KPI):
    """ This class represents a derived KPI which is a ratio of two columns. 
    Names of the the two columns are passed as numerator and denominator.
    
    :param name: name of the kpi
    :type  name: str
    :param numerator: the numerator for the derived KPI
    :type  numerator: str
    :param denominator: the denominator for the derived KPI
    :type  denominator: str
    """
    def __init__(self, name, numerator, denominator):
        super(DerivedKPI, self).__init__(name)
        self.numerator = numerator
        self.denominator = denominator

    def make_derived_kpi(self, data):
        """ Create the derived kpi column if it is not yet created. """
        if self.name not in data.columns:
            data.loc[:, self.name] = (data[self.numerator]/data[self.denominator]).astype("float64")
    
    def get_numerator_and_denominator(self, data):
        """Select two columns from a dataset.
        First corresponds to a numerator of a kpi.
        Second corresponds to a denominator of a kpi. 
        
        :param data: dataframe to extract kpi columns from
        :type  data: pandas.DataFrame

        :return: (numerator, denominator) tuple
        :rtype: (pandas.DataFrame, pandas.DataFrame)
        """
        numerator = data[self.numerator]
        denominator = data[self.denominator]
        return (numerator, denominator)
    
    def exists_in_data(self, data):
        if type(self.numerator) is not str or self.numerator not in data.columns:
            raise KeyError("Numerator '{}' of the derived KPI does not exist in the data.".format(self.numerator))
        if type(self.denominator) is not str or self.denominator not in data.columns:
            raise KeyError("Denominator '{}' of the derived KPI does not exist in the data.".format(self.denominator))
        return True


class StatisticalTestSuite(JsonSerializable):
    """ This class consists of a number of tests plus choice of the correction method.
    
    :param tests: list of statistical tests in the suite
    :type  tests: list[StatisticalTest]
    :param correction_method: method used for multiple testing correction
    :type  correction_method: CorrectionMethod
    """
    def __init__(self, tests, correction_method=CorrectionMethod.NONE):
        if len(tests) is 1:
            correction_method = CorrectionMethod.NONE
        self.tests = tests
        self.correction_method = correction_method

    @property
    def size(self):
        return len(self.tests)


class FeatureFilter(JsonSerializable):
    """ This class represents a filter, restricting a DataFrame to rows with column_value in column_name.
    
    It can be used to specify subgroup conditions.
    :param column_name: name of the column to perform filter on
    :type  column_name: str
    :param column_value: value of the column to perform filter on
    :type  column_value: str
    """
    def __init__(self, column_name, column_value):
        self.column_name  = column_name
        self.column_value = column_value

    def apply_to_data(self, data):
        return data[data[self.column_name] == self.column_value]

    def exists_in_data(self, data):
        if self.column_name not in data.columns:
            raise KeyError("Feature name '{}' does not exist in the data.".format(self.column_name))


class Variants(JsonSerializable):
    """ This class represents information of variants.
    
    :param variant_column_name: name of the column that represents variant
    :type  variant_column_name: str
    :param control_name: value of the variant that represents control group
    :type  control_name: str
    :param treatment_name: value of the variant that represents control group
    :type  treatment_name: str
    """
    def __init__(self, variant_column_name, control_name, treatment_name):
        self.variant_column_name = variant_column_name
        self.control_name        = control_name
        self.treatment_name      = treatment_name

    def get_variant(self, data, variant_name):
        result = data[data[self.variant_column_name] == variant_name]
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame([result])
        return result

    def get_control_variant(self, data):
        """Filter rows in dataframe which belong only to a control variant

        :param data: incoming dataframe to filter
        :type  data: pandas.DataFrame

        :return: subset of rows which correspond only to a control variant
        :rtype: pandas.DataFrame
        """
        return self.get_variant(data, self.control_name)
        
    def get_treatment_variant(self, data):
        """Filter rows in dataframe which belong only to a treatment variant

        :param data: incoming dataframe to filter
        :type  data: pandas.DataFrame

        :return: subset of rows which correspond only to a control variant
        :rtype: pandas.DataFrame
        """
        return self.get_variant(data, self.treatment_name)

    def exists_in_data(self, data):
        if self.variant_column_name not in data.columns:
            raise KeyError("There is no '{}' column in the data.".format(self.variant_column_name))

        return True
        
        
