from expan.core.statistical_test import MultipleTestingCorrectionMethod


# --------- below are the data structure of statistics --------- #
class BaseTestStatistics(object):
    """ holds only statistics (see class SampleStatistics below) for control and treatment """
    def __init__(self, control_statistics, treatment_statistics):
        self.control_statistics   = control_statistics     # of type SampleStatistics
        self.treatment_statistics = treatment_statistics   # of type SampleStatistics


class SampleStatistics(object):
    """ this class holds sample size, mean and variance """
    def __init__(self, sample_size, mean, variance):
        self.sample_size = sample_size    # int
        self.mean        = mean           # float
        self.variance    = variance       # float


class SimpleTestStatistics(BaseTestStatistics):
    """ additionally to BaseTestStatistics, holds delta, confidence interval, statistical power, and p value """
    def __init__(self, control_statistics, treatment_statistics, delta, ci, p, statistical_power):
        super().__init__(control_statistics, treatment_statistics)
        self.delta               = delta              # float
        self.p                   = p                  # float
        self.statistical_power   = statistical_power  # float
        self.confidence_interval = ci                 # of type ConfidenceInterval


class EarlyStoppingTestStatistics(SimpleTestStatistics):
    """ additionally to SimpleTestStatistics, holds boolean early stopping flag """
    def __init__(self, control_statistics, treatment_statistics, delta, ci, p, statistical_power, stop):
        super().__init__(control_statistics, treatment_statistics, delta, ci, p, statistical_power)
        self.stop = stop   # boolean


class CorrectedTestStatistics(object):
    """ additionally to SimpleTestStatistics, holds corrected p-value and corrected confident interval """
    def __init__(self, original_test_statistics, corrected_test_statistics):
        """original_test_statistics and corrected_test_statistics should be instances of
        SimpleTestStatistics or EarlyStoppingTestStatistics"""
        type1 = type(original_test_statistics)
        type2 = type(corrected_test_statistics)
        if type1 != type2:
            raise RuntimeError("Type mismatch for type " + str(type1) + " and " + str(type2))
        self.original_test_statistics  = original_test_statistics
        self.corrected_test_statistics = corrected_test_statistics


class ConfidenceInterval(object):
    """ this class represents the confidence interval"""
    def __init__(self, confidence_interval):
        c_i = [{'percentile': p, 'value': v} for (p, v) in confidence_interval.items()]
        self.confidence_interval = c_i


# --------- below are the data structure of test results --------- #
class StatisticalTestResults(object):
    """ this class holds the results of a single statistical test """
    def __init__(self, test, results):
        self.test    = test          # of type StatisticalTest
        self.results = results       # of type BaseTestStatistics or its subclasses


class MultipleTestSuiteResult(object):
    """ this class holds the results of a MultipleTestSuite """
    def __init__(self, statistical_test_results, correction_method=MultipleTestingCorrectionMethod.no_correction):
        self.statistical_test_results = statistical_test_results   # list of StatisticalTestResults
        self.correction_method        = correction_method          # of type MultipleTestingCorrectionMethod
