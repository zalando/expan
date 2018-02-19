from enum import Enum


class StatisticalTest(object):
    """ this class describes what has to be tested against what and
    represent a unit of statistical testing """
    def __init__(self, kpi_name, features, variants, formula=None):
        self.kpi_name  = kpi_name    # name of kpi
        self.features  = features    # list of type FeatureFilter for features
        self.variants  = variants    # of type Variant
        self.formula   = formula     # formula of the kpi, if this is a derived kpi


class MultipleTestingCorrectionMethod(Enum):
    """ just an enum with three possible correction methods """
    no_correction                 = 0
    bonferroni_correction         = 1
    benjamini_hochberg_correction = 2


class MultipleTestSuite(object):
    """ consists of a number of tests plus choice of the correction method """
    def __init__(self, tests, correction_method=MultipleTestingCorrectionMethod.no_correction):
        self.tests             = tests  # list of type StatisticalTests
        self.correction_method = correction_method

    @property
    def size(self):
        return len(self.tests)


class FeatureFilter(object):
    """ this class represents a filter, restricting a dataframe to rows with column_value in column_name """
    def __init__(self, column_name, column_value):
        self.column_name  = column_name
        self.column_value = column_value


class Variants(object):
    """ this class represents information of variants """
    def __init__(self, variant_column_name, control_name, treatment_name):
        self.variant_column_name = variant_column_name
        self.control_name        = control_name
        self.treatment_name      = treatment_name
