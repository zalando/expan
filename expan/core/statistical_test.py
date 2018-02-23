import pandas as pd
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
        data.loc[:, self.name] = data[self.numerator]/data[self.denominator].astype(float)


class StatisticalTestSuite(JsonSerializable):
    """ This class consists of a number of tests plus choice of the correction method.
    
    :param tests: list of statistical tests in the suite
    :type  tests: list[StatisticalTest]
    :param correction_method: method used for multiple testing correction. Possible values are:
                              "none": no correction
                              "bh": benjamini hochberg correction
                              "bf": bonferroni correction
    :type  correction_method: str
    """
    def __init__(self, tests, correction_method="none"):
        self.tests = tests
        if correction_method not in ["none", "bh", "bf"]:
            raise ValueError('Correction method is not implemented. We support "none", "bh", and "bf".')
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
        return data[self.column_name == self.column_value]


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

    def get_treatment(self, data):
        treatment = data[data.variant_column_name == self.treatment_name]
        if not isinstance(treatment, pd.DataFrame):
            treatment = pd.DataFrame([treatment])
        return treatment

    def get_control(self, data):
        control = data[data.variant_column_name == self.control_name]
        if not isinstance(control, pd.DataFrame):
            control = pd.DataFrame([control])
        return control
