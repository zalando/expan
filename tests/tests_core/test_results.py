# fileencoding: utf8
import json
import os
import re
import unittest
import warnings

import numpy as np

import expan.core.results as results
from expan.core.experiment import Experiment
from expan.core.util import generate_random_data

data_dir = os.getcwd() + '/tests/tests_core/'


# Fixme: depreciated?
def load_example_results():
    """
    This just loads example data so that we need always generate random stuff
    in order to test.

    Also demonstrates loading of HDF5 into expan

    Returns Results object.
    """

    example_fname = 'example_results.h5'
    example_fpath = os.path.join(data_dir, example_fname)
    return results.from_hdf(example_fpath)


def mock_results_object(data, derived_kpis=None, **kwargs):
    """
    Create a results object for any tests involving fixed_horizon_delta()
    """
    res = results.Results(None, metadata=data.metadata)
    res.metadata['reference_kpi'] = {}
    if 'weighted_kpis' in kwargs:
        res.metadata['weighted_kpis'] = kwargs['weighted_kpis']

    pattern = '([a-zA-Z][0-9a-zA-Z_]*)'
    # determine the complete KPI name list
    kpis_to_analyse = data.kpi_names.copy()
    if derived_kpis is not None:
        for dk in derived_kpis:
            kpis_to_analyse.update([dk['name']])
            # assuming the columns in the formula can all be cast into float
            # and create the derived KPI as an additional column
            data.kpis.loc[:, dk['name']] = eval(re.sub(pattern, r'data.kpis.\1.astype(float)', dk['formula']))
            # store the reference metric name to be used in the weighting
            # TODO: only works for ratios
            res.metadata['reference_kpi'][dk['name']] = re.sub(pattern + '/', '', dk['formula'])

    return res


class ResultsTestCase(unittest.TestCase):
    """
    Defines the setUp() and tearDown() functions for the results test cases.
    """

    def setUp(self):
        """
        Load the needed datasets for all TestCases and set the random
        seed so that randomized algorithms show deterministic behaviour.
        """
        np.random.seed(0)
        self.data = Experiment('B', *generate_random_data())
        # Create time column. TODO: Do this nicer
        self.data.kpis['time_since_treatment'] = \
            self.data.features['treatment_start_time']
        # Make time part of index
        self.data.kpis.set_index('time_since_treatment', append=True, inplace=True)

    def tearDown(self):
        """
        Clean up after the test
        """
        # TODO: find out if we have to remove data manually
        pass


class ResultsClassTestCase(ResultsTestCase):
    def testExampleResults(self):
        h5py_available = False
        import imp
        try:
            imp.find_module('h5py')
            imp.find_module('tables')
            h5py_available = True
        except Exception:
            warnings.warn(
                """Could not import h5py or tables module. Skipping
                testExampleResults(). Please make sure that you have the h5py
                and tables packages installed."""
            )

        if h5py_available:
            # aa = load_example_results()
            warnings.warn("No data for h5 loading available... skipping tests of example h5 data")

    def test_relative_uplift_delta(self):
        """Check if the calculation of relative uplift for delta results is
        correct.
        """
        resObj = mock_results_object(self.data)
        res = self.data.fixed_horizon_delta(resObj)
        df = res.relative_uplift('delta', 'normal_same')
        np.testing.assert_almost_equal(df, np.array([[-4.219601, 0]]), decimal=5)

    # def test_prob_uplift_over_zero_single_metric(self):
    # 	"""Check if the conversion from confidence intervals to probability is correct for one metric."""
    # 	res = self.data.delta(kpi_subset=['normal_same'])
    # 	# df = prob_uplift_over_zero_single_metric(res.df, self.data.baseline_variant)
    # 	np.testing.assert_almost_equal(res.df.loc[pd.IndexSlice[:, :, :, 'prob_uplift_over_0'], 'value'],
    # 								   np.array([[0.946519, np.nan]]), decimal=5)
    #
    # def test_prob_uplift_over_zero_multiple_metric(self):
    # 	"""Check if the conversion from confidence intervals to probability is correct for multiple metrics."""
    # 	res = self.data.delta(kpi_subset=['normal_same', 'normal_shifted'])
    # 	# res.calculate_prob_uplift_over_zero()
    # 	np.testing.assert_almost_equal(res.df.loc[pd.IndexSlice[:, :, :, 'prob_uplift_over_0'], 'value'],
    # 								   np.array([[0.946519, np.nan], [0, np.nan]]), decimal=5)

    def test_to_json_delta(self):
        json_object = json.loads(
            self.data.delta(
                kpi_subset=['normal_same'],
                percentiles=[2.5, 97.5]
            ).to_json()
        )
        self.assertEqual(2, len(json_object['variants']))
        self.assertEqual(1, len(json_object['variants'][0]['metrics']))
        self.assertEqual(1, len(json_object['variants'][0]['metrics'][0]['subgroup_metrics']))
        self.assertEqual(1, len(json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups']))
        self.assertEqual(4, len(
            json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups'][0]['statistics']))
        self.assertEqual(1, len(
            json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups'][0]['statistics'][3][
                'pctiles']))

    def test_to_json_delta_to_file(self):
        self.data.delta(
            kpi_subset=['normal_same'],
            percentiles=[2.5, 97.5]
        ).to_json(fpath="test_json.json")

        with open("test_json.json", 'r') as json_file:
            json_object = json.load(fp=json_file)

        self.assertEqual(2, len(json_object['variants']))
        self.assertEqual(1, len(json_object['variants'][0]['metrics']))
        self.assertEqual(1, len(json_object['variants'][0]['metrics'][0]['subgroup_metrics']))
        self.assertEqual(1, len(json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups']))
        self.assertEqual(4, len(
            json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups'][0]['statistics']))
        self.assertEqual(1, len(
            json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups'][0]['statistics'][3][
                'pctiles']))

        os.remove("test_json.json")

    def test_to_json_sga(self):
        json_object = json.loads(
            self.data.sga(
                percentiles=[2.5, 5.0, 95.0, 97.5]
            ).to_json()
        )
        self.assertEqual(2, len(json_object['variants']))
        self.assertEqual(4, len(json_object['variants'][0]['metrics']))
        self.assertEqual(2, len(json_object['variants'][0]['metrics'][0]['subgroup_metrics']))
        self.assertGreaterEqual(4, len(json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups']))
        self.assertEqual(4, len(
            json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups'][0]['statistics']))
        self.assertEqual(1, len(
            json_object['variants'][0]['metrics'][0]['subgroup_metrics'][0]['subgroups'][0]['statistics'][3][
                'pctiles']))

    def test_to_json_trend(self):
        # to_json() doesn't handle trend() results yet!
        self.assertIsNone(self.data.trend().to_json())


if __name__ == '__main__':
    # unittest.main()
    np.random.seed(0)
    exp = Experiment('B', *generate_random_data())
    res = exp.delta(['normal_shifted'])
