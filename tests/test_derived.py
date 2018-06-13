import expan.core.early_stopping as es
import expan.core.statistics as statx
from expan.core.experiment import Experiment

import pytest

import numpy as np

worker_table = Experiment(None).worker_table

def tuple_approx(*elements):
    assert isinstance(elements, tuple)

    def helper_for__tuple_approx(x):
        if not hasattr(x, '__len__'):
            return pytest.approx(x)
        else:
            assert isinstance(x, tuple)
            return tuple(helper_for__tuple_approx(element) for element in x)

    return helper_for__tuple_approx(elements)

def deltastats_to_friendly_tuple(ds):
    from collections import namedtuple
    Flat_delta_stats = namedtuple('Flat_delta_stats', 'delta p power stop control_stats treatment_stats c_i')
    tup = Flat_delta_stats(    ds.delta,
                ds.p,
                ds.statistical_power,
                ds.stop if hasattr(ds,'stop') else None,
                (ds.control_statistics  .mean,ds.control_statistics  .sample_size,ds.control_statistics  .variance),
                (ds.treatment_statistics.mean,ds.treatment_statistics.sample_size,ds.treatment_statistics.variance),
                tuple(sorted(map(lambda _ : (_['percentile'],_['value']), ds.confidence_interval))),
            )
    return tup

def very_compact_test(  expected_results,
                        method, extra_args_for_method, extra_kwargs_for_method,
                        x, y,
                        x_denominators=None, y_denominators=None,
                        ):
    assert isinstance(expected_results, tuple)
    worker_factory = worker_table[method]
    worker = worker_factory(*extra_args_for_method, **extra_kwargs_for_method)

    if x_denominators is None:
        assert y_denominators is None
        res = worker(x,y)
    else:
        assert y_denominators is not None
        res = worker(x,y,x_denominators,y_denominators)
    tup = deltastats_to_friendly_tuple(res)
    assert tup == tuple_approx(*expected_results)

def test_derived_fixed_horizon_equalweights():
    very_compact_test(          (-1.0,0.207999999,0.322771,None,(3.0,3,0.6666666)
                                ,(2.0,3,0.66666666),((2.5,-2.8509634),(97.5,0.8509634))),
                                'fixed_horizon',
                                [], {'min_observations':0},
                                np.array([1,2,3]),
                                np.array([2,3,4]),
                                np.array([1,1,1]),
                                np.array([1,1,1]),
                                )

def test_derived_fixed_horizon_equalweights_not_1():
    very_compact_test(          (-1.0,0.207999999,0.322771,None,(3.0,3,0.6666666)
                                ,(2.0,3,0.66666666),((2.5,-2.8509634),(97.5,0.8509634))),
                                'fixed_horizon',
                                [], {'min_observations':0},
                                np.array([1,2,3]),
                                np.array([4,6,8]),
                                np.array([1,1,1]),
                                np.array([2,2,2]),
                                )

def test_derived_fixed_horizon_almost_same_ratio():
    very_compact_test(          (0.0,1.0,0.025,None,(1.0,3,0)
                                ,(1.0,3,0),((2.5,0),(97.5,0))),
                                'fixed_horizon',
                                [], {'min_observations':0},
                                np.array([1,2,3]),
                                np.array([1,2,3]),
                                np.array([1,2,3]),
                                np.array([1,2,3]),
                                )

def test_derived_fixed_horizon_almost_same_ratio_morevariety():
    very_compact_test(          (0.0,1.0,0.025,None,(1.0,3,0)
                                ,(1.0,3,0),((2.5,0),(97.5,0))),
                                'fixed_horizon',
                                [], {'min_observations':0},
                                np.array([2,1,3]),
                                np.array([1,4,6]),
                                np.array([2,1,3]),
                                np.array([1,4,6]),
                                )

def test_derived_fixed_horizon_almost_x_ratio_diff_from_y_ratio():
    very_compact_test(          (1.0,5e-12,0.4164563,None,(1.0,3,0)
                                ,(2.0,3,0),((2.5,1),(97.5,1))),
                                'fixed_horizon',
                                [], {'min_observations':0},
                                np.array([4,2,6]),
                                np.array([1,4,6]),
                                np.array([2,1,3]),
                                np.array([1,4,6]),
                                )
