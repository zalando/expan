from enum import Enum

import pandas as pd
from expan.core.util import JsonSerializable

from copy import deepcopy


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


class CorrectionMethod(Enum):
    """ Correction methods."""
    NONE       = 1   # no correction
    BONFERRONI = 2   # Bonferroni correction. Used to correct false positive rate.
    BH         = 3   # Benjamini-Hochberg procedure. Used to correct false discovery rate.


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
