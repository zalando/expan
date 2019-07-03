from __future__ import division
from expan.core.experiment import Experiment, _choose_threshold_type
import pandas as pd
import numpy as np


def test_choose_threshold_type_upper():
    data = np.array([1, 2, 3, np.nan])
    assert _choose_threshold_type(data) == 'upper'


def test_choose_threshold_type_upper_zero():
    data = np.array([0, 1, 2, 3, np.nan])
    assert _choose_threshold_type(data) == 'upper'


def test_choose_threshold_type_lower():
    data = np.array([-3, -2, -1, np.nan])
    assert _choose_threshold_type(data) == 'lower'


def test_choose_threshold_type_lower_zero():
    data = np.array([-3, -2, -1, 0, np.nan])
    assert _choose_threshold_type(data) == 'lower'


def test_choose_threshold_type_two_sided():
    data = np.array([-3, -2, -1, 0, 1, 2, 3, np.nan])
    assert _choose_threshold_type(data) == 'two-sided'


def test_quantile_filtering_upper_old():
    exp = Experiment({})
    data = np.array([0,0,1,2]) / np.array([0,0,1,1])
    df = pd.DataFrame.from_dict({'earnings' : data})

    flags = exp._quantile_filtering(df, ['earnings'], {'earnings': ('upper', 90.0)})
    assert flags.tolist() == [False, False, False, True]


def test_quantile_filtering_lower_old():
    exp = Experiment({})
    data = np.array([0,0,1,2]) / np.array([0,0,1,1])
    df = pd.DataFrame.from_dict({'earnings' : data})

    flags = exp._quantile_filtering(df, ['earnings'], {'earnings': ('lower', 10.)})
    assert flags.tolist() == [False, False, True, False]


def test_quantile_filtering_upper():
    exp = Experiment({})
    data = np.array([0.0]*2 + list(range(10))) / np.array([0.0]*2 + [1.0]*10)
    df = pd.DataFrame.from_dict({'earnings' : data})

    flags = exp._quantile_filtering(df, ['earnings'], {'earnings': ('upper', 90.0)})
    assert flags.tolist() == [False]*11 + [True]


def test_quantile_filtering_lower():
    exp = Experiment({})
    data = np.array([0.0]*2 + list(range(10))) / np.array([0.0]*2 + [1.0]*10)
    df = pd.DataFrame.from_dict({'earnings' : data})

    flags = exp._quantile_filtering(df, ['earnings'], {'earnings': ('lower', 50.0)})
    print(flags.tolist())
    assert flags.tolist() == [False]*2 + [True]*5 + [False]*5


def test_quantile_filtering_two_sided():
    exp = Experiment({})
    df = pd.DataFrame.from_dict({'earnings' : list(range(10))})

    flags = exp._quantile_filtering(df, ['earnings'], {'earnings': ('two-sided', 80.0)})
    results = flags.tolist()
    assert results == [True] + [False]*8 + [True]

def test_quantile_filtering_two_sided_asym():
    exp = Experiment({})
    data = list(range(-8,0)) + list(range(16))
    df = pd.DataFrame.from_dict({'earnings' : data})

    flags = exp._quantile_filtering(df, ['earnings'],
                                    {'earnings': ('two-sided-asym', 50.0)})
    results = flags.tolist()
    assert results == [True]*2 + [False]*18 + [True]*4
