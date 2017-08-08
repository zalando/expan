import unittest

import numpy as np
import pandas as pd

from expan.core.experiment import Experiment
# from expan.core.results import Results
from expan.core.util import generate_random_data, get_column_names_by_type, find_list_of_dicts_element
# from tests.tests_core.test_results import mock_results_object

class ExperimentTestCase(unittest.TestCase):
    """
    Defines the setUp() and tearDown() functions for the statistics test cases.
    """

    def setUp(self):
        """
        Load the needed datasets for all StatisticsTestCases and set the random
        seed so that randomized algorithms show deterministic behaviour.
        """
        np.random.seed(0)
        data, metadata = generate_random_data()
        self.column_names = list(set(data.columns) - set(['variant', 'entity']))
        self.numeric_column_names = get_column_names_by_type(data, np.number)

        self.data, self.metadata = data, metadata

    def tearDown(self):
        """
        Clean up after the test
        """
        # TODO: find out if we have to remove data manually
        pass


class ExperimentClassTestCases(ExperimentTestCase):
    """
    Test cases for the Experiment class functions.
    """

    # valid ones
    derived_kpi_1 = {'name'   : 'derived_kpi_1',
                     'formula': 'normal_same/normal_shifted'}

    derived_kpi_2 = {'name'   : 'derived_kpi_2',
                     'formula': 'normal_shifted/normal_same'}

    # bad ones
    derived_kpi_3 = {'name'   : 'derived_kpi_3',
                     'formula': 'normal_shifted/non_existing'}

    derived_kpi_4 = {'name'   : 'derived_kpi_4',
                     'formula': 'non_existing/normal_same'}

    # badly structured cases
    # not a dictionary
    derived_kpi_5_1,  derived_kpi_5_2 = ['name', 'derived_kpi_5'], ['formula', 'normal_same/normal_same']
    derived_kpi_6 = ['name', 'derived_kpi_6']

    # do not have proper keys
    derived_kpi_7 = {'name': 'derived_kpi_7'}
    derived_kpi_8 = {'name_': 'derived_kpi_8',
                     'formula_': 'normal_shifted/normal_same'}
    derived_kpi_9 = {'derived_kpi_8': 'normal_shifted/normal_same'}


    def assertNumericalEqual(self, a, b, decimals):
        self.assertEqual(round(a, decimals), round(b, decimals))


    def getExperiment(self, report_kpi_names=[], derived_kpis=[]):
        return Experiment('B', self.data, self.metadata, report_kpi_names, derived_kpis)


    def test_constructor(self):
        self.getExperiment()

        with self.assertRaises(ValueError):
            self.getExperiment(self.column_names + ['non_existing'])

        self.getExperiment(self.column_names + [self.derived_kpi_1['name'],
                                                self.derived_kpi_2['name']],
                          [self.derived_kpi_1, self.derived_kpi_2])

        with self.assertRaises(ValueError):
            self.getExperiment(self.column_names + [self.derived_kpi_1['name'],
                                                    self.derived_kpi_3['name']],
                               [self.derived_kpi_1, self.derived_kpi_3])


        with self.assertRaises(ValueError):
            self.getExperiment(self.column_names + [self.derived_kpi_4['name'],
                                                    self.derived_kpi_2['name']],
                               [self.derived_kpi_4, self.derived_kpi_2])

        with self.assertRaises(TypeError):
            self.getExperiment(123)

        self.getExperiment(['normal_same'])

        # implicit do the conversion if there is one str
        self.getExperiment('normal_same')

        # check of dictionary structure
        with self.assertRaises(TypeError):
            self.getExperiment(self.numeric_column_names + ['normal_same'],
                               [self.derived_kpi_5_1, self.derived_kpi_5_2])

        with self.assertRaises(TypeError):
            self.getExperiment(self.numeric_column_names + ['normal_same'],
                               [self.derived_kpi_6])

        with self.assertRaises(KeyError):
            self.getExperiment(self.numeric_column_names + [self.derived_kpi_7['name'], 'normal_same'],
                               [self.derived_kpi_7])

        with self.assertRaises(KeyError):
            self.getExperiment(self.numeric_column_names + [self.derived_kpi_1['name'], 'normal_same'],
                               [self.derived_kpi_1, self.derived_kpi_8])

        with self.assertRaises(KeyError):
            self.getExperiment(self.numeric_column_names + ['normal_same'],
                               [self.derived_kpi_9])

    def test_errors_warnings_expan_version(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='fixed_horizon')
        a = 'errors'        in res
        a = 'warnings'      in res
        a = 'expan_version' in res


    def test_fixed_horizon_delta(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='fixed_horizon')

        variants = find_list_of_dicts_element(res['kpis'], 'name', 'normal_same', 'variants')
        aStats   = find_list_of_dicts_element(variants, 'name', 'A', 'delta_statistics')

        self.assertNumericalEqual(aStats['delta'],           0.033053, ndecimals)

        self.assertNumericalEqual(aStats['confidence_interval'][0]['value'], -0.007135, ndecimals)
        self.assertNumericalEqual(aStats['confidence_interval'][1]['value'],  0.073240, ndecimals)

        self.assertEqual(aStats['treatment_sample_size'], 6108)
        self.assertEqual(aStats['control_sample_size'],   3892)

        self.assertNumericalEqual(aStats['treatment_mean'],  0.025219, ndecimals)
        self.assertNumericalEqual(aStats['control_mean'],   -0.007833, ndecimals)


    def test_fixed_horizon_delta_derived_kpis(self):
        self.getExperiment(self.numeric_column_names + [self.derived_kpi_1['name'],
                                                      self.derived_kpi_2['name']],
                           [self.derived_kpi_1, self.derived_kpi_2]).delta()


    def test_group_sequential_delta(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='group_sequential')

        variants = find_list_of_dicts_element(res['kpis'], 'name', 'normal_same', 'variants')
        aStats   = find_list_of_dicts_element(variants, 'name', 'A', 'delta_statistics')
        self.assertNumericalEqual(aStats['delta'],           0.033053, ndecimals)

        self.assertNumericalEqual(aStats['confidence_interval'][0]['value'], -0.007135, ndecimals)
        self.assertNumericalEqual(aStats['confidence_interval'][1]['value'],  0.073240, ndecimals)

        self.assertEqual(aStats['treatment_sample_size'], 6108)
        self.assertEqual(aStats['control_sample_size'],   3892)

        self.assertNumericalEqual(aStats['treatment_mean'],  0.025219, ndecimals)
        self.assertNumericalEqual(aStats['control_mean'],   -0.007833, ndecimals)


    def test_group_sequential_delta_derived_kpis(self):
        self.getExperiment(self.numeric_column_names + [self.derived_kpi_1['name'],
                                                      self.derived_kpi_2['name']],
                           [self.derived_kpi_1, self.derived_kpi_2]).delta('group_sequential')

    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor_delta(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='bayes_factor', num_iters=2000)

        variants = find_list_of_dicts_element(res['kpis'], 'name', 'normal_same', 'variants')
        aStats   = find_list_of_dicts_element(variants, 'name', 'A', 'delta_statistics')
        self.assertNumericalEqual(aStats['delta'], 0.033053, ndecimals)

        self.assertEqual(aStats['stop'],      True, ndecimals)
        self.assertEqual(aStats['number_of_iterations'], 2000, ndecimals)

        #
        # this can result in different numerical values depending on Python version
        #
        # self.assertNumericalEqual(aStats['confidence_interval'][0]['value'], -0.007079081, ndecimals)
        # self.assertNumericalEqual(aStats['confidence_interval'][1]['value'],  0.072703576, ndecimals)

        self.assertEqual(aStats['treatment_sample_size'], 6108)
        self.assertEqual(aStats['control_sample_size'],   3892)

        self.assertNumericalEqual(aStats['treatment_mean'],  0.025219, ndecimals)
        self.assertNumericalEqual(aStats['control_mean'],   -0.007833, ndecimals)


    # @unittest.skip("sometimes takes too much time")
    def test_bayes_factor_delta_derived_kpis(self):
        exp = self.getExperiment([self.derived_kpi_1['name']], [self.derived_kpi_1])
        res = exp.delta(method='bayes_factor', num_iters=2000)


    # @unittest.skip("sometimes takes too much time")
    def test_bayes_precision_delta(self):
        ndecimals = 5
        res = self.getExperiment(['normal_same']).delta(method='bayes_precision', num_iters=2000)

        variants = find_list_of_dicts_element(res['kpis'], 'name', 'normal_same', 'variants')
        aStats   = find_list_of_dicts_element(variants, 'name', 'A', 'delta_statistics')
        self.assertNumericalEqual(aStats['delta'], 0.033053, ndecimals)

        self.assertEqual(aStats['stop'], True, ndecimals)
        self.assertEqual(aStats['number_of_iterations'], 2000, ndecimals)

        #
        # this can result in different numerical values depending on Python version
        #
        # self.assertNumericalEqual(aStats['confidence_interval'][0]['value'], -0.007079081, ndecimals)
        # self.assertNumericalEqual(aStats['confidence_interval'][1]['value'],  0.072703576, ndecimals)

        self.assertEqual(aStats['treatment_sample_size'], 6108)
        self.assertEqual(aStats['control_sample_size'],   3892)

        self.assertNumericalEqual(aStats['treatment_mean'],  0.025219, ndecimals)
        self.assertNumericalEqual(aStats['control_mean'],   -0.007833, ndecimals)


    # @unittest.skip("sometimes takes too much time")
    def test_bayes_precision_delta_derived_kpis(self):
        exp = self.getExperiment([self.derived_kpi_1['name']], [self.derived_kpi_1])
        res = exp.delta(method='bayes_precision', num_iters=2000)

        # self.getExperiment([self.derived_kpi_1['name']], [self.derived_kpi_1]).delta()


if __name__ == '__main__':
    unittest.main()
