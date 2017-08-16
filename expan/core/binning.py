import logging
import string
import warnings

import numpy as np

from expan.core.util import is_number_and_nan

logger = logging.getLogger(__name__)



#------- bin representations -------#
class Bin(object):
    def __init__(self, id, bin_type, representation):
        '''
        Constructor for a bin object.
        :param id: identifier (e.g. bin number) of the bin
        :param bin_type: "numerical" or "categorical"
        :param representation: representation of the bin. depends on the bin type.
        '''
        if bin_type == "numerical" and type(representation) is not NumericalRepresenation:
            raise ValueError("Use NumericalRepresenation object to initialize a numerical bin.")
        if bin_type == "categorical" and type(representation) is not CategoricalRepresenation:
            raise ValueError("Use CategoricalRepresenation object to initialize a categorical bin.")
        self.id = id
        self.bin_type = bin_type
        self.representation = representation

    def __repr__(self):
        return "\nbin id: " + str(self.id) + ", bin: " + str(self.representation)


class NumericalRepresenation(object):
    def __init__(self, lower, upper, lower_closed, upper_closed):
        '''
        Constructor for representation of a numerical bin.
        :param upper: upper bound of the bin (exclusive)
        :param lower: lower bound of the bin (inclusive)
        :param lower_closed: boolean indicator whether lower bound is closed
        :param upper_closed: boolean indicator whether upper bound is closed
        '''
        self.lower = lower
        self.upper = upper
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed

    def __repr__(self):
        repr = ""
        if self.lower_closed:
            repr += "["
        else:
            repr += "("

        repr += str(self.lower) + ", " + str(self.upper)

        if self.upper_closed:
            repr += "]"
        else:
            repr += ")"
        return repr


class CategoricalRepresenation(object):
    def __init__(self, values):
        '''
        Constructor for representation of a categorical bin.
        :param values: list of categorical values that belong to this bin
        '''
        self.values = values

    def __repr__(self):
        return str(self.values)



#------- public methods -------#
def create_bins(data, n_bins):
    '''
    Create bins from the data value
    :param data: a list or a 1-dim array of data to determine the bins
    :param n_bins: number of bins to create
    :return: a list of Bin object
    '''
    if data is None or len(data) <= 0:
        raise ValueError('Empty input array!')

    if n_bins <= 0:
        raise ValueError('Less than one bin makes no sense.')

    insufficient_distinct = False
    n_unique_values = len(np.unique([value for value in data if not is_number_and_nan(value)]))
    if n_unique_values < n_bins:
        insufficient_distinct = True
        warnings.warn("Insufficient unique values for requested number of bins. " +
                      "Number of bins will be reset to number of unique values.")
        n_bins = n_unique_values

    # cast into a numpy array to infer the dtype
    data_as_array = np.array(data)
    is_numeric = np.issubdtype(data_as_array.dtype, np.number)

    if is_numeric:
        bins = _create_numerical_bins(data_as_array, n_bins)
    else:
        bins = _create_categorical_bins(data_as_array, n_bins)

    if (not insufficient_distinct) and (len(bins) < n_bins):
        warnings.warn('Created less bins than requested.')

    return bins


def assign_bins(data, bins):
    '''
    Assign each data point to one of the bins.
    :param data: a list or a 1-dim array of data to be labeled
    :param bins: a list of Bin object
    :return: a dataframe that associates each data point to a bin.
    '''
    # cast into a numpy array to infer the dtype
    data_as_array = np.array(data)
    is_numeric = np.issubdtype(data_as_array.dtype, np.number)

    if is_numeric:
        labeled_data = _assign_numerical_bins(data_as_array, bins)
    else:
        labeled_data = _assign_categorical_bins(data_as_array, bins)
    return labeled_data



#------- private methods for numerical binnings-------#

def _create_numerical_bins(data_as_array, n_bins):
    return _create_next_numerical_bin(data_as_array, n_bins, 0, [])

def _create_next_numerical_bin(x, n_bins, id, result):
    '''
    Create bins for numerical data
    :param x: array of data
    :param n_bins: number of bins
    :param id: id of the next bin
    :return: a list of bins object
    '''
    return

def _assign_numerical_bins(data_as_array, bins):
    return



#------- private methods for categorical binnings-------#
def _create_categorical_bins(data_as_array, n_bins):
    return

def _assign_categorical_bins(data_as_array, bins):
    return
