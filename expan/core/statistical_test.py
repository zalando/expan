import re
from enum import Enum

from expan.core.util import JsonSerializable


class StatisticalTest(JsonSerializable):
    """ This class describes what has to be tested against what and represent a unit of statistical testing. 
    :param kpi: the kpi to perform on
    :type  kpi: KPI or its subclass
    :param features: list of features used for subgroups
    :type  features: list[FeatureFilter]
    :param variants: variant column name and their values
    :type  variants: Variants
    """
    def __init__(self, kpi, features, variants):
        self.kpi       = kpi
        self.features  = features
        self.variants  = variants


class KPI(JsonSerializable):
    """ This class represents a basic kpi.
    :param name: name of the kpi
    :type  name: str
    """
    def __init__(self, name):
        self.name = name


class DerivedKPI(KPI):
    """ This class represents a derived KPI which is a ratio of two columns.
    :param name: name of the kpi
    :type  name: str
    :param formula: formula of the kpi. It should be a ratio of two column. e.g. 'revenue/session'
    :type  formula: str
    """
    def __init__(self, name, formula):
        super(DerivedKPI, self).__init__(name)
        self.formula = formula
        self.reference_kpi = re.sub('([a-zA-Z][0-9a-zA-Z_]*)/', '', formula)


class MultipleTestingCorrectionMethod(Enum):
    """ Enum with three possible correction methods. """
    no_correction                 = 0
    bonferroni_correction         = 1
    benjamini_hochberg_correction = 2


class StatisticalTestSuite(JsonSerializable):
    """ This class consists of a number of tests plus choice of the correction method. 
    :param tests: list of statistical tests in the suite
    :type  tests: list[StatisticalTest]
    :param correction_method: method used for multiple testing correction
    :type  correction_method: MultipleTestingCorrectionMethod
    """
    def __init__(self, tests, correction_method=MultipleTestingCorrectionMethod.no_correction):
        self.tests             = tests
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
