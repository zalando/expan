""" NB: This module is deprecated. """

import logging
import warnings
from heapq import heapify, heappush, heappop

import numpy as np

from expan.core.util import is_nan

logger = logging.getLogger(__name__)


#------- bin representations -------#
class Bin(object):
    # this is a necessary hack for the buggy implementation of assertItemsEqual in python2
    # see https://stackoverflow.com/a/29690198
    # note that if we only use python3, assertCountEqual in python3 solves this problem
    __hash__ = None

    def __init__(self, bin_type, *repr_args):
        """
        Constructor for a bin object.
        :param id: identifier (e.g. bin number) of the bin
        :param bin_type: "numerical" or "categorical"
        :param repr_args: arguments to represent this bin. 
        args for numerical bin includes lower, upper, lower_closed, upper_closed
        args for categorical bin includes a list of categories for this bin.
        """
        if bin_type == "numerical" and len(repr_args) != 4:
            raise ValueError("args for numerical bin are lower, upper, lower_closed, upper_closed.")
        if bin_type == "categorical" and len(repr_args) != 1 and type(repr_args[0]) is not list:
            raise ValueError("args for categorical bin is a list of categorical values for this bin.")
        self.bin_type = bin_type

        if bin_type == "numerical":
            self.representation = NumericalRepresentation(*repr_args)
        elif bin_type == "categorical":
            self.representation = CategoricalRepresentation(*repr_args)

    def __repr__(self):
        return "\nbin: " + str(self.representation)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, data, feature):
        """
        Apply the bin to data.
        :param data: pandas data frame
        :param feature: feature name on which this bin is defined
        :return: subset of input dataframe which belongs to this bin
        """
        return self.representation.apply_to_data(data, feature)


class NumericalRepresentation(object):
    # this is a necessary hack for the buggy implementation of assertItemsEqual in python2
    # see https://stackoverflow.com/a/29690198
    # note that if we only use python3, assertCountEqual in python3 solves this problem
    __hash__ = None

    def __init__(self, lower, upper, lower_closed, upper_closed):
        """
        Constructor for representation of a numerical bin.
        :param upper: upper bound of the bin (exclusive)
        :param lower: lower bound of the bin (inclusive)
        :param lower_closed: boolean indicator whether lower bound is closed
        :param upper_closed: boolean indicator whether upper bound is closed
        """
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

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_to_data(self, data, feature):
        """
        Apply the bin to data.
        :param data: pandas data frame
        :param feature: feature name on which this bin is defined
        :return: subset of input dataframe which belongs to this bin
        """
        data_feature_column = data[feature]

        # if either bound is nan, only nans exist in the bin.
        if np.isnan(self.lower) or np.isnan(self.upper):
            return data[np.isnan(data_feature_column)]

        if self.lower_closed:
            filter_lower = (data_feature_column >= self.lower)
        else:
            filter_lower = (data_feature_column > self.lower)
        if self.upper_closed:
            filter_upper = (data_feature_column <= self.upper)
        else:
            filter_upper = (data_feature_column < self.upper)

        return data[filter_lower & filter_upper]


class CategoricalRepresentation(object):
    # this is a necessary hack for the buggy implementation of assertItemsEqual in python2
    # see https://stackoverflow.com/a/29690198
    # note that if we only use python3, assertCountEqual in python3 solves this problem
    __hash__ = None

    def __init__(self, categories):
        """
        Constructor for representation of a categorical bin.
        :param categories: list of categorical values that belong to this bin
        """
        try:
            categories = set(categories)
        except:
            raise ValueError("categorical bin should be represented by an iterable object of categories.")

        self.categories = categories

    def __repr__(self):
        return str(list(self.categories))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_to_data(self, data, feature):
        """
        Apply the bin to data.
        :param data: pandas data frame
        :param feature: feature name on which this bin is defined
        :return: subset of input dataframe which belongs to this bin
        """
        data_feature_column = data[feature]
        return data[data_feature_column.isin(self.categories)]



#------- public methods -------#
def create_bins(data, n_bins):
    """
    Create bins from the data value
    :param data: a list or a 1-dim array of data to determine the bins
    :param n_bins: number of bins to create
    :return: a list of Bin object
    """
    if data is None or len(data) <= 0:
        raise ValueError('Empty input array!')

    if n_bins <= 0:
        raise ValueError('Less than one bin makes no sense.')

    insufficient_distinct = False
    n_unique_values = len(np.unique([value for value in data if not is_nan(value)]))
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


#------- private methods for numerical binnings-------#

def _create_numerical_bins(data_as_array, n_bins):
    return _create_next_numerical_bin(data_as_array, n_bins, [])


def _create_next_numerical_bin(x, n_bins, result):
    """
    Create bins for numerical data
    :param x: array of data
    :param n_bins: number of bins
    :return: a list of bins object
    """
    # no more data
    if len(x) == 0:
        return result

    # if data has nan
    if any(np.isnan(x)):
        cur_bin = Bin("numerical", np.nan, np.nan, True, True)
        result.append(cur_bin)
        return _create_next_numerical_bin(x[~np.isnan(x)], n_bins-1, result)

    # the last bin is a closed-closed interval
    if n_bins == 1:
        cur_bin = Bin("numerical", min(x), max(x), True, True)
        result.append(cur_bin)
        return result

    lower, upper, lower_closed, upper_closed = _first_interval(x, n_bins)
    cur_bin = Bin("numerical", lower, upper, lower_closed, upper_closed)
    result.append(cur_bin)

    next_data = x[x > upper] if upper_closed else x[x >= upper]
    return _create_next_numerical_bin(next_data, n_bins-1, result)


def _first_interval(x, n_bins):
    """
    Gets the first interval based on the percentiles, 
    either a closed interval containing the same value multiple times
    or a closed-open interval with a different lower and upper bound.
    """
    # calculate the percentiles
    percentiles = np.linspace(0., 100., n_bins + 1)
    bounds = np.percentile(x, q=percentiles, interpolation='higher')
    lower = bounds[0]
    upper = bounds[1]

    if lower == upper:
        return lower, upper, True, True
    else:
        return lower, upper, True, False


#------- private methods for categorical binnings-------#

def _create_categorical_bins(data_as_array, n_bins):
    """ 
    Performs greedy (non-optimal) binning
    :param data_as_array: array of data to bin according to their frequencies
    :return: a list of Bin object
    """
    # count items
    weights = {}
    for item in data_as_array:
        if not item in weights:
            weights[item] = 1
        else:
            weights[item] += 1

    # we need items sorted in decreasing order
    pairs = sorted([(weight, [item]) for (item, weight) in weights.items()], reverse=True)
    bins = pairs[:min(n_bins, len(pairs))]

    # too little data, just return what we have so far
    if len(pairs) <= n_bins:
        return toBinObject(bins)

    heapify(bins)

    # go through pairs, from heaviest to lightest
    for (pair_weight, pair_labels) in pairs[n_bins:]:
        # take the lightest bin
        bin_weight, bin_labels = heappop(bins)
        # add the heaviest item to it
        heappush(bins, (bin_weight + pair_weight, bin_labels + pair_labels))

    return toBinObject(bins)


def toBinObject(bins):
    result = []
    for bin in bins:
        categories = bin[1]
        result.append(Bin("categorical", categories))
    return result
