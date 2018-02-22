from expan.core.util import JsonSerializable


# --------- Below are the data structure of statistics --------- #
class BaseTestStatistics(JsonSerializable):
    """ Holds only statistics for the control and treatment group. 
    :param control_statistics: statistics within the control group
    :type  control_statistics: SampleStatistics
    :param treatment_statistics: statistics within the treatment group
    :type  treatment_statistics: SampleStatistics
    """
    def __init__(self, control_statistics, treatment_statistics):
        self.control_statistics   = control_statistics
        self.treatment_statistics = treatment_statistics


class SampleStatistics(JsonSerializable):
    """ This class holds sample size, mean and variance.
    :type sample_size: int
    :type mean: float
    :type variance: float
    """
    def __init__(self, sample_size, mean, variance):
        self.sample_size = sample_size
        self.mean        = mean
        self.variance    = variance


class SimpleTestStatistics(BaseTestStatistics):
    """ Additionally to BaseTestStatistics, holds delta, confidence interval, statistical power, and p value.
    :type control_statistics: SampleStatistics
    :type treatment_statistics: SampleStatistics
    :type delta: float
    :type p: float
    :type statistical_power: float
    :param ci: a dict where keys are percentiles and values are the corresponding value for the statistic.
    :type  ci: dict
    """
    def __init__(self, control_statistics, treatment_statistics, delta, ci, p, statistical_power):
        super(SimpleTestStatistics, self).__init__(control_statistics, treatment_statistics)
        self.delta               = delta
        self.p                   = p
        self.statistical_power   = statistical_power
        self.confidence_interval = [{'percentile': p, 'value': v} for (p, v) in ci.items()]


class EarlyStoppingTestStatistics(SimpleTestStatistics):
    """ Additionally to SimpleTestStatistics, holds boolean flag for early stopping.
    :type control_statistics: SampleStatistics
    :type treatment_statistics: SampleStatistics
    :type delta: float
    :type p: float
    :type statistical_power: float
    :param ci: a dict where keys are percentiles and values are the corresponding value for the statistic.
    :type  ci: dict
    :type stop: bool
    """
    def __init__(self, control_statistics, treatment_statistics, delta, ci, p, statistical_power, stop):
        super(EarlyStoppingTestStatistics, self).__init__(control_statistics, treatment_statistics, delta, ci, p, statistical_power)
        self.stop = stop


class CorrectedTestStatistics(JsonSerializable):
    """ Holds original and corrected statistics. This class should be used to hold statistics for multiple testing.
    original_test_statistics and corrected_test_statistics should have the same type.
    :param original_test_statistics: test result before correction
    :type  original_test_statistics: SimpleTestStatistics or EarlyStoppingTestStatistics
    :param corrected_test_statistics: test result after correction
    :type  corrected_test_statistics: SimpleTestStatistics or EarlyStoppingTestStatistics
    """
    def __init__(self, original_test_statistics, corrected_test_statistics):
        type1 = type(original_test_statistics)
        type2 = type(corrected_test_statistics)
        if type1 != type2:
            raise RuntimeError("Type mismatch for type " + str(type1) + " and " + str(type2))
        if not isinstance(original_test_statistics, BaseTestStatistics):
            raise RuntimeError("Input should be instances of BaseTestStatistics or its subclass")
        self.original_test_statistics  = original_test_statistics
        self.corrected_test_statistics = corrected_test_statistics


# --------- Below are the data structure of test results --------- #
class StatisticalTestResult(JsonSerializable):
    """ This class holds the results of a single statistical test.
    :param test: information about the statistical test
    :type  test: StatisticalTest
    :param result: result of this statistical test
    :type  result: BaseTestStatistics or its subclasses or CorrectedTestStatistics  #TODO: better approach?
    """
    def __init__(self, test, result):
        self.test   = test
        self.result = result


class MultipleTestSuiteResult(JsonSerializable):
    """ This class holds the results of a MultipleTestSuite.
    :param statistical_test_results: test results for all statistical testing unit
    :type  statistical_test_results: list[StatisticalTestResult]
    :param correction_method: method used for multiple testing correction. Possible values are:
                              "none": no correction
                              "bh": benjamini hochberg correction
                              "bf": bonferroni correction
    :type  correction_method: str
    """
    def __init__(self, statistical_test_results, correction_method="none"):
        self.statistical_test_results = statistical_test_results
        if correction_method not in ["none", "bh", "bf"]:
            raise ValueError('Correction method is not implemented. We support "none", "bh", and "bf".')
        self.correction_method = correction_method
