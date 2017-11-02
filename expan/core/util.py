from warnings import warn

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def find_list_of_dicts_element(items, key1, value, key2):
    return [item[key2] for item in items if item[key1] == value][0]

def is_number_and_nan(obj):
    return obj != obj

def get_column_names_by_type(df, dtype):
    return [c for c in df.columns if np.issubdtype(df.dtypes[c], dtype)]

def drop_nan(np_array):
    if np_array.ndim == 1:
        return np_array[~np.isnan(np_array)]
    elif np_array.ndim == 2:
        return np_array[~np.isnan(np_array).any(axis=1)]


def scale_range(x, new_min=0.0, new_max=1.0, old_min=None, old_max=None, squash_outside_range=True, squash_inf=False, ):
    """
    Scales a sequence to fit within a new range.

    If squash_inf is set, then infinite values will take on the
    extremes of the new range (as opposed to staying infinite).

    Args:
        x:
        new_min:
        new_max:
        old_min:
        old_max:
        squash_outside_range:
        squash_inf:

    Note:
        Infinity in the input is disregarded in the construction of the scale of the mapping.

  >>> scale_range([1,3,5])
  array([ 0. ,  0.5,  1. ])

  >>> scale_range([1,2,3,4,5])
  array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

  >>> scale_range([1,3,5, np.inf])
  array([ 0. ,  0.5,  1. ,  inf])

  >>> scale_range([1,3,5, -np.inf])
  array([ 0. ,  0.5,  1. , -inf])

  >>> scale_range([1,3,5, -np.inf], squash_inf=True)
  array([ 0. ,  0.5,  1. ,  0. ])

  >>> scale_range([1,3,5, np.inf], squash_inf=True)
  array([ 0. ,  0.5,  1. ,  1. ])

  >>> scale_range([1,3,5], new_min=0.5)
  array([ 0.5 ,  0.75,  1.  ])

  >>> scale_range([1,3,5], old_min=1, old_max=4)
  array([ 0.        ,  0.66666667,  1.        ])

  >>> scale_range([5], old_max=4)
  array([ 1.])

  """
    if squash_inf and not squash_outside_range:
        warn(ValueError('Makes no sense to squash infinity but not other numbers outside the source range. \
         Will squash all outside range.'))
        squash_outside_range = True

    if isinstance(x, list):
        x = np.array(x)

    if old_min is None:
        old_min = np.min(x[~np.isinf(x)])
    if old_max is None:
        old_max = np.max(x[~np.isinf(x)])
    old_range = old_max - old_min
    new_max = float(new_max)
    new_min = float(new_min)
    new_range = new_max - new_min

    retval = (new_range * (x - old_min) / old_range) + new_min
    if squash_inf:
        retval[np.isinf(x) & (x > 0)] = new_max
        retval[np.isinf(x) & (x < 0)] = new_min

    if squash_outside_range:
        retval[~np.isinf(x) & (x > old_max)] = new_max
        retval[~np.isinf(x) & (x < old_min)] = new_min

    return retval


def generate_random_data():
    np.random.seed(42)
    size = 10000

    test_data_frame = pd.DataFrame()
    test_data_frame['entity'] = list(range(size))
    test_data_frame['variant'] = np.random.choice(['A', 'B'], size=size, p=[0.6, 0.4])

    test_data_frame['normal_same'] = np.random.normal(size=size)
    test_data_frame['normal_shifted'] = np.random.normal(size=size)

    size_shifted_B = test_data_frame['normal_shifted'][test_data_frame['variant'] == 'B'].shape[0]
    test_data_frame.loc[test_data_frame['variant'] == 'B', 'normal_shifted'] \
        = np.random.normal(loc=1.0, size=size_shifted_B)

    test_data_frame['feature'] = np.random.choice(['has', 'non'], size=size)
    test_data_frame['normal_shifted_by_feature'] = np.random.normal(size=size)

    ii = (test_data_frame['variant'] == 'B') & (test_data_frame['feature'] == 'has')
    randdata_shifted_mean = np.random.normal(loc=1.0, size=sum(ii == True))
    test_data_frame.loc[ii, 'normal_shifted_by_feature'] = randdata_shifted_mean

    test_data_frame['treatment_start_time'] = np.random.choice(list(range(10)), size=size)
    test_data_frame['normal_unequal_variance'] = np.random.normal(size=size)

    size_unequalvar_B = test_data_frame['normal_unequal_variance'][test_data_frame['variant'] == 'B'].shape[0]
    test_data_frame.loc[test_data_frame['variant'] == 'B','normal_unequal_variance'] \
        = np.random.normal(scale=10, size=size_unequalvar_B)

    # Add date column
    d1 = datetime.strptime('2015-01-01', '%Y-%m-%d')
    d2 = datetime.strptime('2016-03-01', '%Y-%m-%d')
    date_col = []

    delta = d2 - d1
    for i in range(delta.days * 24 + 1):
        date_col.append((d1 + timedelta(hours=i)).strftime('%Y-%m-%d'))

    test_data_frame['date'] = date_col[:size]

    metadata = {
        'primary_KPI': 'normal_shifted',
        'source': 'simulated',
        'experiment': 'random_data_generation'
    }

    return test_data_frame, metadata


def generate_random_data_n_variants(n_variants=3):
    np.random.seed(42)
    size = 10000

    test_data_frame = pd.DataFrame()
    test_data_frame['entity'] = list(range(size))
    test_data_frame['variant'] = np.random.choice(list(map(chr, list(range(65, 65 + n_variants)))), size=size)

    test_data_frame['normal_same'] = np.random.normal(size=size)
    test_data_frame['poisson_same'] = np.random.poisson(size=size)

    test_data_frame['feature'] = np.random.choice(['has', 'non'], size=size)

    test_data_frame['treatment_start_time'] = np.random.choice(list(range(10)), size=size)

    metadata = {
        'primary_KPI': 'normal_same',
        'source': 'simulated',
        'experiment': 'random_data_generation'
    }

    return test_data_frame, metadata

