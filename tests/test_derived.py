import expan.core.early_stopping as es
import expan.core.statistics as statx

import pytest

import numpy as np

worker_table = {
    'fixed_horizon'    : statx.make_delta,
    'group_sequential' : es.make_group_sequential,
    'bayes_factor'     : es.make_bayes_factor,
    'bayes_precision'  : es.make_bayes_precision
}

def dict_multilookup(d):
    t = ( d.delta,
                    d.control_statistics.sample_size, d.treatment_statistics.sample_size,
                    d.control_statistics.mean, d.treatment_statistics.mean,
                    d.control_statistics.variance, d.treatment_statistics.variance,
                    d.p
                    )
    return t

def test_derived_fixed_horizon_equalweights():
    worker = worker_table['fixed_horizon']    (   min_observations=0  )
    x = np.array([1,2,3])
    y = np.array([2,3,4])
    ds = worker(x,y
            , x_denominators=np.array([1,1,1])
            , y_denominators=np.array([1,1,1])
            )
    assert dict_multilookup(ds)  \
            == pytest.approx((-1,
                3,3,
                3,2,
                0.66666666,0.66666666,
                0.20799999999999982,
                ))
    c_i = sorted(map(lambda _ : sorted(_.items()), ds.confidence_interval))
    assert pytest.approx(c_i) == [[('percentile', 2.5), ('value', -2.8509634034651996)], [('percentile', 97.5), ('value', 0.8509634034651992)]]

def test_derived_fixed_horizon_equalweights_not_1():
    # by doubling the numerator and denominator throughout
    # y, nothing changes in the result. As expected
    worker = worker_table['fixed_horizon']    (   min_observations=0  )
    x = np.array([1,2,3])
    y = np.array([4,6,8])
    ds = worker(x,y
            , x_denominators=np.array([1,1,1])
            , y_denominators=np.array([2,2,2])
            )
    assert dict_multilookup(ds)  \
            == pytest.approx((-1,
                3,3,
                3,2,
                0.66666666,0.66666666,
                0.20799999999999982,
                ))
    c_i = sorted(map(lambda _ : sorted(_.items()), ds.confidence_interval))
    assert pytest.approx(c_i) == [[('percentile', 2.5), ('value', -2.8509634034651996)], [('percentile', 97.5), ('value', 0.8509634034651992)]]

def test_derived_fixed_horizon_almost_same_ratio():
    # by doubling the numerator and denominator throughout
    # y, nothing changes in the result. As expected
    worker = worker_table['fixed_horizon']    (   min_observations=0  )
    x = np.array([1,2,3])
    y = np.array([1,2,3])
    ds = worker(x,y
            , x_denominators=np.array([1,2,3])
            , y_denominators=np.array([1,2,3])
            )
    some = dict_multilookup(ds)
    assert some == pytest.approx((0,
                3,3,
                1,1,
                0,0,
                1.0,
                ))
    c_i = sorted(map(lambda _ : sorted(_.items()), ds.confidence_interval))
    assert c_i == [[('percentile', 2.5), ('value', pytest.approx(-0.0))], [('percentile', 97.5), ('value', pytest.approx(0.0))]]

def test_derived_fixed_horizon_almost_same_ratio_morevariety():
    # by doubling the numerator and denominator throughout
    # y, nothing changes in the result. As expected
    worker = worker_table['fixed_horizon']    (   min_observations=0  )
    x = np.array([2,1,3])
    y = np.array([1,4,6])
    ds = worker(x,y
            , x_denominators=np.array([2,1,3])
            , y_denominators=np.array([1,4,6])
            )
    some = dict_multilookup(ds)
    assert some == pytest.approx((0,
                3,3,
                1,1,
                0,0,
                1.0,
                ))
    c_i = sorted(map(lambda _ : sorted(_.items()), ds.confidence_interval))
    assert c_i == [[('percentile', 2.5), ('value', pytest.approx(-0.0))], [('percentile', 97.5), ('value', pytest.approx(0.0))]]

def test_derived_fixed_horizon_almost_x_ratio_diff_from_y_ratio():
    # by doubling the numerator and denominator throughout
    # y, nothing changes in the result. As expected
    worker = worker_table['fixed_horizon']    (   min_observations=0  )
    x = np.array([4,2,6])
    y = np.array([1,4,6])
    ds = worker(x,y
            , x_denominators=np.array([2,1,3])
            , y_denominators=np.array([1,4,6])
            )
    some = dict_multilookup(ds)
    assert some == pytest.approx((1,
                3,3,
                1,2,
                0,0,
                5.999960000210015e-12,
                ))
    c_i = sorted(map(lambda _ : sorted(_.items()), ds.confidence_interval))
    assert c_i == [[('percentile', 2.5), ('value', pytest.approx(1))], [('percentile', 97.5), ('value', pytest.approx(1))]]
