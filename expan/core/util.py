import json
import logging
from enum import Enum

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class JsonSerializable(object):
    """ Interface for serializable classes."""
    def toJson(self):
        return json.dumps(self, default=lambda o: o.name if isinstance(o, Enum) else o.__dict__, sort_keys=True, indent=4)

    def __repr__(self):
        return self.toJson()


def find_value_by_key_with_condition(items, condition_key, condition_value, lookup_key):
    """ Find the value of lookup key where the dictionary contains condition key = condition value.
    
    :param items: list of dictionaries
    :type  items: list
    :param condition_key: condition key
    :type  condition_key: str
    :param condition_value: a value for the condition key
    :param lookup_key: lookup key or key you want to find the value for
    :type  lookup_key: str
    
    :return: lookup value or found value for the lookup key
    """
    return [item[lookup_key] for item in items if item[condition_key] == condition_value][0]


def is_nan(obj):
    """ Checks whether the input is NaN. It uses the trick that NaN is not equal to NaN."""
    return obj != obj


def drop_nan(array):
    """ Drop Nan values from the given numpy array.
    
    :param array: input array
    :type  array: np.ndarray
    
    :return: a new array without NaN values
    :rtype: np.ndarray
    """
    if array.ndim == 1:
        return array[~np.isnan(array)]
    elif array.ndim == 2:
        return array[~np.isnan(array).any(axis=1)]


def generate_random_data():
    """ Generate random data for two variants. It can be used in unit tests or demo. """
    np.random.seed(42)
    size = 10000

    data = pd.DataFrame()
    data['entity'] = list(range(size))
    data['variant'] = np.random.choice(['A', 'B'], size=size, p=[0.6, 0.4])

    data['normal_same'] = np.random.normal(size=size)
    data['normal_shifted'] = np.random.normal(size=size)

    size_shifted_B = data['normal_shifted'][data['variant'] == 'B'].shape[0]
    data.loc[data['variant'] == 'B', 'normal_shifted'] = np.random.normal(loc=1.0, size=size_shifted_B)

    data['feature'] = np.random.choice(['has', 'non'], size=size)
    data.loc[0, 'feature'] = 'feature that only has one data point'
    data['normal_shifted_by_feature'] = np.random.normal(size=size)

    ii = (data['variant'] == 'B') & (data['feature'] == 'has')
    randdata_shifted_mean = np.random.normal(loc=1.0, size=sum(ii == True))
    data.loc[ii, 'normal_shifted_by_feature'] = randdata_shifted_mean

    data['treatment_start_time'] = np.random.choice(list(range(10)), size=size)
    data['normal_unequal_variance'] = np.random.normal(size=size)

    size_unequalvar_B = data['normal_unequal_variance'][data['variant'] == 'B'].shape[0]
    data.loc[data['variant'] == 'B', 'normal_unequal_variance'] = np.random.normal(scale=10, size=size_unequalvar_B)

    # Add date column
    d1 = datetime.strptime('2015-01-01', '%Y-%m-%d')
    d2 = datetime.strptime('2016-03-01', '%Y-%m-%d')
    date_col = []
    delta = d2 - d1
    for i in range(delta.days * 24 + 1):
        date_col.append((d1 + timedelta(hours=i)).strftime('%Y-%m-%d'))
    data['date'] = date_col[:size]

    metadata = {
        'primary_KPI': 'normal_shifted',
        'source': 'simulated',
        'experiment': 'random_data_generation'
    }
    return data, metadata
