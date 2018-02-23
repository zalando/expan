import json
import logging
from enum import Enum

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class JsonSerializable(object):
    def toJson(self):
        return json.dumps(self, default=lambda o: o.name if isinstance(o, Enum) else o.__dict__, sort_keys=True, indent=4)

    def __repr__(self):
        return self.toJson()


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
    test_data_frame.loc[0, 'feature'] = 'feature that only has one data point'
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
