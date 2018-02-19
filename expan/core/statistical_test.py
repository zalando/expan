from enum import Enum


class StatisticalTest(object):
    """ This class describes what has to be tested against what and represent a unit of statistical testing. 
    :param kpi_name: name of the kpi
    :type  kpi_name: str
    :param features: list of features used for subgroups
    :type  features: list[FeatureFilter]
    :param variants: variant column name and their values
    :type  variants: Variants
    :param formula: formula of the kpi, if this is a derived kpi
    :type  formula: str
    """
    def __init__(self, kpi_name, features, variants, formula=None):
        self.kpi_name  = kpi_name
        self.features  = features
        self.variants  = variants
        self.formula   = formula


class MultipleTestingCorrectionMethod(Enum):
    """ Enum with three possible correction methods. """
    no_correction                 = 0
    bonferroni_correction         = 1
    benjamini_hochberg_correction = 2


class MultipleTestSuite(object):
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


class FeatureFilter(object):
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


class Variants(object):
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
