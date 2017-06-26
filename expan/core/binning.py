# fileencoding: utf8

import logging
import string
import warnings

import numpy as np

from expan.core.util import is_number_and_nan

symbol_universal_set = 'ξ'  # https://en.wikipedia.org/wiki/Xi_(letter)
logger = logging.getLogger(__name__)

class Binning(object):
    def __init__(self):
        """
        The Binning class has two subclasses: CategoricalBinning and NumericalBinning.
        """
        pass

    def __len__(self):
        raise NotImplementedError('must implement this in subclass')

    def label(self, data, format_str=None):
        """
        This returns the bin labels associated with each data point in the series,
        essentially 'applying' the binning to data.

        Args:
            data (array-like): array of datapoints to be binned
            format_str (str): string defining the format of the label to apply

        Returns:
            array-like: array of the bin label corresponding to each data point.

        Note:
            Implemented in subclass.
        """
        raise NotImplementedError('must implement this in subclass')

    def __str__(self, format_str='{standard}'):
        res = "{} with {:d} bins:".format(self.__class__.__name__, len(self))
        for ii, lbl in enumerate(self.labels(format_str)):
            res += '\n {:d}: {}'.format(ii, lbl)
        return res


class CategoricalBinning(Binning):
    """
    A CategoricalBinning is essentially a list of lists of categories. 
    Each bin within a Binning is an ordered list of categories.
    """

    def __init__(self, data=None, nbins=None):
        """CategoricalBinning constructor

        Args:
            data (array-like): array of datapoints to be binned
            nbins (int): number of bins
        """
        super(CategoricalBinning, self).__init__()
        if data is None:
            # NB: this will go through the property setters, so they will not be simply lists.
            self.categories = []
        else:
            self.categories = self._get_binning_categorical(data, nbins)

    def _get_binning_categorical(self, x, n_bins):
        """
        Generates a dictionary of Interval objects for categorical variables,
        divides into the same number of bins as the number of unique categories.

        Args:
            x (array_like): input array

        Returns:
            interval_dict (dict): dictionary with the key-value pair {name : Interval}

        """
        # group NAs into a single interval, the effective n_bins will be increased by 1
        if any([is_number_and_nan(xx) for xx in x]):
            rest_list = self._get_binning_categorical([xx for xx in x if not is_number_and_nan(xx)], n_bins)
            return rest_list + [[np.nan]]

        levels = np.unique(x)
        cat_list = []
        ssize_list = []

        for ll in levels:
            cat_list.append([ll])
            ssize_list.append(sum([xx == ll for xx in x]))

        # greedy approach to merge bins if needed
        if n_bins is not None and len(levels) > n_bins:
            while len(cat_list) != n_bins:
                cat_list, ssize_list = self._merge_two_smallest_intervals(cat_list, ssize_list)

        return cat_list

    def _merge_two_smallest_intervals(self, category_list, sample_size_list):

        assert len(category_list) == len(sample_size_list), 'category_list and sample_size_list of unequal lengths!'

        pairs = list(zip(sample_size_list, category_list))  # Schwartzian transform aka decorate-sort-undecorate
        pairs.sort()

        ssl_and_cl = list(zip(*pairs))  # unzip
        ssl = list(ssl_and_cl[0])
        cl = list(ssl_and_cl[1])

        ssl[1] = ssl[0] + ssl[1]  # merge [0] + [1] -> [1]
        cl[1] = cl[0] + cl[1]

        del (ssl[0])  # drop the smallest one
        del (cl[0])

        return cl, ssl

    @property
    def categories(self):
        """
        Returns list of categories.

        Returns:
            array-like: list of categories
        """
        return self._categories[:-1]

    def __len__(self):
        return len(self._categories) - 1

    @categories.setter
    def categories(self, value):
        self._categories = value + [np.inf]  # represent the universal set like this?

    @property
    def _mids(self):
        """
        The middle category of every bin
        """
        return np.array(
            [None if len(c) == 0 else c[len(c) // 2] for c in self.categories]
        )

    def _labels(self, format_str='{standard}'):
        """
        The components of the format_str recognised here are:
            - {iter.uppercase}, {iter.lowercase}, {iter.integer} -
            - {set_notation} - all categories, comma-separated, surrounded by curly braces
            - {standard} - a shortcut for: {set_notation}
        """

        lbls = []
        it = None
        format_str = format_str.replace('{standard}', '{set_notation}')

        if '{iter.' in format_str:
            if '{iter.uppercase}' in format_str:
                it = iter(string.ascii_uppercase)
            elif '{iter.lowercase}' in format_str:
                it = iter(string.ascii_lowercase)
            elif '{iter.integer}' in format_str:
                import itertools
                it = itertools.count()
            else:
                raise NotImplementedError('Unknown iterator in format_str ({})'.format(
                    format_str))

            format_str = format_str.replace(
                '{iter.uppercase}', '{iterator}').replace(
                '{iter.lowercase}', '{iterator}').replace(
                '{iter.integer}', '{iterator}')

        for ii, (cats) in enumerate(self._categories):

            is_catchall = ii == len(self)

            format_args = {
                'set_notation': '{' + ','.join([str(c) for c in cats]) + '}' if
                cats != np.inf else symbol_universal_set,
            }
            if is_catchall:
                format_args['set_notation'] = '{unseen}'

            if it is not None:
                if not is_catchall:
                    format_args['iterator'] = next(it)
                else:
                    format_args['iterator'] = '?'

            lbl = format_str.format(**format_args)
            lbls += [lbl]

        return np.array(lbls)

    def labels(self, format_str='{standard}'):
        """
        Returns the labels of the bins defined by this binning.

        Args:
            format_str (str): string defining the format of the label to return

                Options:

                * {iter.uppercase}, {iter.lowercase}, {iter.integer} 
                * {set_notation} - all categories, comma-separated, surrounded by curly braces
                * {standard} - a shortcut for: {set_notation}

        Returns:
            array-like: labels of the bins defined by this binning

        Note:
            This is not the same as label (which applies the bins to data and returns the labels of the data).
        """
        # NB: the last item in self._labels is the universal set symbol
        # TODO: do we really need the universal set symbol?
        return self._labels(format_str)[: -1]

    def _apply(self, data):
        """
        This applies the binning to a data vector, returning the indices into the
        categories list. Underlies all the label functions.

        Currently iterates through all bins regardless of match, and thus
        the latest bins that match will take precedence.
        """
        try:
            out = np.empty(data.shape, int)
        except AttributeError:  # duck typing
            # data = np.array(data)
            # do NOT cast data into numpy arrays, otherwise NaNs will be treated
            # as strings
            out = np.empty(len(data), int)

        logger.debug('apply: _categories: ' + str(self._categories))

        out[:] = -1
        for ii, (cats) in enumerate(self.categories):
            # NB: in1d converts inputs to arrays, so strings will be
            # compared as strings etc.
            # and nans, if numeric array, will not properly be detected.
            # got  = np.in1d(data, cats)
            got = np.array([d in cats for d in data])

            logger.debug('apply: testing {:d} {}: {:d} members'.format(ii, self._labels('{standard}')[ii], got.sum()))
            out[got] = ii

        return out

    def mid(self, data):
        """
        Returns the middle category of every bin.

        Args:
            data: data on which the binning is to be applied

        Returns:
            array-like: the middle category of every bin
        """
        return self._mids[self._apply(data)]

    def label(self, data, format_str='{standard}'):
        """
        This returns the bin labels associated with each data point in the series,
        essentially 'applying' the binning to data.

        Args:
            data (array-like): array of datapoints to be binned
            format_str (str): string defining the format of the label to apply

                Options:

                * {iter.uppercase}, {iter.lowercase}, {iter.integer} 
                * {set_notation} - all categories, comma-separated, surrounded by curly braces
                * {standard} - a shortcut for: {set_notation}

        Returns:
            array-like: array of the bin label corresponding to each data point
        """
        if format_str is None:  # TODO: remove this special case?
            lbls = self.mid(data)
        else:
            bin_lbls = self._labels(format_str)
            lbls = bin_lbls[self._apply(data)]

        return lbls


class NumericalBinning(Binning):
    """
    The Binning class for numerical variables.

    Todo:
        Think of a good way of exposing the _apply() method, because with the returned indices, 
        it can then get uppers/lowers/mids/labels (ie reformat) without doing the apply again.
        And experimenting with maintaining the lists with a single element tacked onto the end representing non-matching entries.

        All access then are through properties which drop this end list, 
        except when using the indices returned by _apply.

        This means that the -1 indices just works, so using the indices to get labels, bounds, etc., 
        is straightforward and fast because it is just integer-based array slicing.
    """
    def __init__(self, data=None, nbins=None, uppers=None, lowers=None,
                 up_closed=None, lo_closed=None):
        """
        NumericalBinning constructor.

        Args:
            data (array-like): array of datapoints to be binned
            nbins (int): number of bins
            uppers (array-like): a list of upper bounds
            lowers (array-like): a list of lower bounds
            up_closed (array-like): a list of booleans indicating whether the 
                upper bounds are closed
            lo_closed (array-like): a list of booleans indicating whether the 
                lower bounds are closed
        """
        super(NumericalBinning, self).__init__()
        if data is None:
            # initialize binning explicitly?
            if (uppers is not None and lowers is not None and
                        up_closed is not None and lo_closed is not None):
                self.uppers = uppers
                self.lowers = lowers
                self.up_closed = up_closed
                self.lo_closed = lo_closed
            else:
                # NB: this will go through the property setters, so they will not be
                # simply lists.
                self.uppers = []
                self.lowers = []
                self.up_closed = []
                self.lo_closed = []
        else:
            # TODO: this is extremely simple, but doesn't handle skewed distributions
            # properly. Update with the interval-based code.
            # percentiles = np.linspace(0., 100., nbins + 1)  # [1:]
            # upper = np.percentile(data[~np.isnan(data)], q=percentiles,
            # 		interpolation='higher')

            self.lowers, self.uppers, self.lo_closed, self.up_closed = self._get_binning_numeric_recursive(data, nbins)

            # self.lowers = upper[:-1]
            # self.uppers = upper[1:]

            # self.lo_closed=~(self.lowers==np.NaN) #will always be true
            # self.up_closed=(self.uppers==np.NaN) #will always be false

            # #The last bin is considered closed on its upper bound.
            # self.up_closed[-1]=True

    def _get_binning_numeric_recursive(self, x, n_bins, open_ends=False):
        """
        get_binning for numeric variables. All Intervals are ClosedOpenInterval
        except the last one containing the maximum, which is then a ClosedClosedInterval.
        """
        # group NAs into a single interval, the effective n_bins will be increased by 1
        if any(np.isnan(x)):
            na_lower = [np.nan]
            na_upper = [np.nan]
            na_lo_closed = [True]
            na_up_closed = [True]
            lower, upper, lo_closed, up_closed = self._get_binning_numeric_recursive(x[~np.isnan(x)], n_bins)
            na_lower += lower
            na_upper += upper
            na_lo_closed += lo_closed
            na_up_closed += up_closed
            return na_lower, na_upper, na_lo_closed, na_up_closed

        # no more data
        if len(x) == 0:
            return [], [], [], []

        # the last bin is a closed-closed interval
        if n_bins == 1:
            return [min(x)], [max(x)], [True], [True]

        first_lower, first_upper, first_lo_closed, first_up_closed = self._first_interval(x, n_bins)
        if n_bins != 1 and first_lo_closed[0] and not first_up_closed[0]:
            # previous intervals are ClosedOpenInterval
            next_lower, next_upper, next_lo_closed, next_up_closed = self._get_binning_numeric_recursive(
                x[x >= first_upper[0]], n_bins - 1)
        else:
            # the last interval is ClosedClosedInterval
            next_lower, next_upper, next_lo_closed, next_up_closed = self._get_binning_numeric_recursive(
                x[x > first_upper[0]], n_bins - 1)

        first_lower += next_lower
        first_upper += next_upper
        first_lo_closed += next_lo_closed
        first_up_closed += next_up_closed
        return first_lower, first_upper, first_lo_closed, first_up_closed

    def _first_interval(self, x, n_bins):
        """
        Gets the first interval based on the percentiles, either a
        ClosedClosedInterval containing the same value multiple times or a
        ClosedOpenInterval with a different lower and upper bound.
        """
        # calculate the percentiles
        percentiles = np.linspace(0., 100., n_bins + 1)
        bounds = np.percentile(x, q=percentiles, interpolation='higher')
        lower = bounds[0]
        upper = bounds[1]

        if lower == upper:
            # closed-closed
            return [lower], [upper], [True], [True]
        else:
            # closed-open
            return [lower], [upper], [True], [False]

    @property
    def uppers(self):
        """Return a list of upper bounds."""
        return self._uppers[:-1]

    @property
    def lowers(self):
        """Return a list of lower bounds."""
        return self._lowers[:-1]

    @property
    def up_closed(self):
        """Return a list of booleans indicating whether the upper bounds are 
        closed.
        """
        return self._up_closed[:-1]

    @property
    def lo_closed(self):
        """Return a list of booleans indicating whether the lower bounds are 
        closed."""
        return self._lo_closed[:-1]

    @uppers.setter
    def uppers(self, value):
        self._uppers = np.hstack((value, [np.nan]))  # this effectively coerces to float array

    @lowers.setter
    def lowers(self, value):
        self._lowers = np.hstack((value, [np.nan]))  # this effectively coerces to float array

    @up_closed.setter
    def up_closed(self, value):
        self._up_closed = np.hstack((value, [False]))

    @lo_closed.setter
    def lo_closed(self, value):
        self._lo_closed = np.hstack((value, [True]))

    def _labels(self, format_str='{standard}'):
        """
        This returns an array of the bin labels (same length as uppers) formatted
        according to the provided string.

        As much as possible, the formatting is left to the standard string format,
        allowing precision to be specified for numerical values, strings to be
        concatenated, etc...

        The components of the format_str recognised here are:
            - {iter.uppercase} and {iter.lowercase} = labels the bins with letters
            - {iter.integer} = labels the bins with integers
            - {up} and {lo} = the bounds themselves (can specify precision: {up:.1f})
            - {up_cond} and {lo_cond} = '<', '<=' etc.
            - {up_bracket} and {lo_bracket} = '(', '[' etc.
            - {mid}       = the midpoint of the bin (can specify precision: {mid:.1f}
            - {conditions} = {lo:.1f}{lo_cond}x{up_cond}{up:.1f}
            - {set_notation} = {lo_bracket}{lo:.1f},{up:.1f}{up_bracket}
            - {standard} = {conditions}
            - {simple} = {lo:.1f}_{up:.1f}
            - {simplei} = {lo:.0f}_{up:.0f} (same as simple but for integers)
        """
        # TODO: has been implemented in an ugly way, but the interface should be good
        lbls = []
        it = None

        orig_format_str = format_str
        format_str = orig_format_str.replace(
            '{standard}', '{set_notation}').replace(
            '{conditions}', '{lo:.1f}{lo_cond}x{up_cond}{up:.1f}').replace(
            '{set_notation}', '{lo_bracket}{lo:.1f},{up:.1f}{up_bracket}').replace(
            '{simple}', '{lo:.1f}_{up:.1f}').replace(
            '{simplei}', '{lo:.0f}_{up:.0f}')
        catchall_format_str = orig_format_str.replace(
            '{standard}', '{set_notation}').replace(
            '{conditions}', 'unseen').replace(
            '{set_notation}', '[unseen]').replace(
            '{simple}', 'unseen').replace(
            '{simplei}', 'unseen')

        if '{iter.' in format_str:
            if '{iter.uppercase}' in format_str:
                it = iter(string.ascii_uppercase)
            elif '{iter.lowercase}' in format_str:
                it = iter(string.ascii_lowercase)
            elif '{iter.integer}' in format_str:
                import itertools
                it = itertools.count()
            else:
                raise NotImplementedError('Unknown iterator in format_str ({})'.format(
                    format_str))

            format_str = format_str.replace(
                '{iter.uppercase}', '{iterator}').replace(
                '{iter.lowercase}', '{iterator}').replace(
                '{iter.integer}', '{iterator}')
            catchall_format_str = catchall_format_str.replace(
                '{iter.uppercase}', '?').replace(
                '{iter.lowercase}', '?').replace(
                '{iter.integer}', '?')

        for ii, (lc, uc, l, u, m) in enumerate(zip(self._lo_closed,
                                                   self._up_closed,
                                                   self._lowers,
                                                   self._uppers,
                                                   self._mids)):

            is_catchall = ii == len(self)
            format_args = {
                'lo': l,
                'up': u,
                'mid': m,
                'lo_cond': '<=' if lc else '<',
                'up_cond': '<=' if uc else '<',
                'lo_bracket': '[' if lc else '(',
                'up_bracket': ']' if uc else ')',
            }
            if (not is_catchall) and (it is not None):
                format_args['iterator'] = next(it)

            lbl = (format_str if not is_catchall else catchall_format_str).format(**format_args)
            lbls += [lbl]

        return np.array(lbls)

    def labels(self, format_str='{standard}'):
        """
        Returns the labels of the bins defined by this binning.

        Returns:
            array-like: array of labels of the bins defined by this binning

                Options:

                - {iter.uppercase} and {iter.lowercase} = labels the bins with letters
                - {iter.integer} = labels the bins with integers
                - {up} and {lo} = the bounds themselves (can specify precision: {up:.1f})
                - {up_cond} and {lo_cond} = '<', '<=' etc.
                - {up_bracket} and {lo_bracket} = '(', '[' etc.
                - {mid}       = the midpoint of the bin (can specify precision: {mid:.1f}
                - {conditions} = {lo:.1f}{lo_cond}x{up_cond}{up:.1f}
                - {set_notation} = {lo_bracket}{lo:.1f},{up:.1f}{up_bracket}
                - {standard} = {conditions}
                - {simple} = {lo:.1f}_{up:.1f}
                - {simplei} = {lo:.0f}_{up:.0f} (same as simple but for integers)

        Note:
            This is not the same as label (which applies the bins to data and returns the labels of the data)
        """
        return self._labels(format_str)[0: -1]

    @property
    def _mids(self):
        return (self._uppers + self._lowers) / 2.

    def __len__(self):
        return len(self._uppers) - 1

    def _apply(self, data):
        """
        This applies the binning to a data vector, returning the indices into the
        uppers/lowers etc. lists. Underlies all the label functions.

        Currently iterates through all bins regardless of match, and thus
        the latest bins that match will take precedence.
        """
        data = np.array(data)

        try:
            out = np.empty(data.shape, int)
        except AttributeError:  # duck typing
            data = np.array(data)
            out = np.empty(data.shape, int)

        logger.debug('apply: _uppers: ' + str(self._uppers))
        logger.debug('apply: _lowers: ' + str(self._lowers))
        logger.debug('apply: _mids: ' + str(self._mids))
        logger.debug('apply: _up_closed: ' + str(self._up_closed))
        logger.debug('apply: _lo_closed: ' + str(self._lo_closed))

        out[:] = -1
        for ii, (lc, uc, l, u) in enumerate(
            zip(self.lo_closed, self.up_closed, self.lowers, self.uppers)):
            if np.isnan(l) or np.isnan(u):  # if either bound is nan, only nans exist in the bin.
                got = np.isnan(data)
            else:
                got = (l <= data) if lc else (l < data)
                got &= (data <= u) if uc else (data < u)

            logger.debug('apply: testing {:d} {}: {:d} members'.format(ii, self._labels('{conditions}')[ii], got.sum()))
            out[got] = ii

        return out

    def _bound(self, data, which):
        if which == 'upper':
            bounds = self._uppers
        elif which == 'lower':
            bounds = self._lowers
        elif which == 'mid':
            bounds = self._mids
        else:
            raise ValueError('Unknown bound \'{}\'.'.format(which))

        return np.array(bounds)[self._apply(data)]

    def upper(self, data):
        """
        Returns the upper bounds of the bins associated with the data

        Arguments:
            data (array-like): array of datapoints to be binned

        Returns:
            array-like: array containing upper bound of bin
                    corresponding to each data point.
        """
        return self._bound(data, 'upper')

    def lower(self, data):
        """ see upper() """
        return self._bound(data, 'lower')

    def mid(self, data):
        """
        Returns the midpoints of the bins associated with the data

        Arguments:
            data (array-like): array of datapoints to be binned

        Returns:
            array-like: array containing midpoints of bin
                    corresponding to each data point.
        
        Note:
            Currently doesn't take into account whether bounds are closed or open.
        """
        return self._bound(data, 'mid')

    def label(self, data, format_str='{standard}'):
        """
        Returns the bin labels associated with each data point in the series,
        essentially 'applying' the binning to data.

        Args:
            data (array-like): array of datapoints to be binned
            format_str (str): string defining the format of the label to apply

                Options:

                - {iter.uppercase} and {iter.lowercase} = labels the bins with letters
                - {iter.integer} = labels the bins with integers
                - {up} and {lo} = the bounds themselves (can specify precision: {up:.1f})
                - {up_cond} and {lo_cond} = '<', '<=' etc.
                - {up_bracket} and {lo_bracket} = '(', '[' etc.
                - {mid}       = the midpoint of the bin (can specify precision: {mid:.1f}
                - {conditions} = {lo:.1f}{lo_cond}x{up_cond}{up:.1f}
                - {set_notation} = {lo_bracket}{lo:.1f},{up:.1f}{up_bracket}
                - {standard} = {conditions}
                - {simple} = {lo:.1f}_{up:.1f}
                - {simplei} = {lo:.0f}_{up:.0f} (same as simple but for integers)

        see:
            - Binning.label.__doc__
            - NumericalBinning._labels.__doc__

        When format_str is None, the label is the midpoint of the bin.
        This may not be the most convenient. Might be better to make
        the default format_str '{standard}' and then have the client use mid()
        directly if midpoints are desired.

        """
        if format_str is None:
            # return the bin index
            lbls = self._apply(data)
        else:
            bin_lbls = self._labels(format_str)
            lbls = bin_lbls[self._apply(data)]

        return lbls


def create_binning(x, nbins=8):
    """
    Determines bins for the input values - suitable for doing SubGroup Analyses.

    Arguments:
        x (array_like): input array
        nbins (integer): number of bins

    Returns:
        binning object
    """
    if x is None or len(x) <= 0:
        raise ValueError('Empty input array!')

    if nbins <= 0:
        raise ValueError('Less than one bin makes no sense.')

    insufficient_distinct = False
    nunique = len(np.unique([xx for xx in x if not is_number_and_nan(xx)]))
    if nunique < nbins:
        insufficient_distinct = True
        warnings.warn('Insufficient distinct values for requested number of bins.')
        nbins = nunique

    # cast into a dummy numpy array to infer the dtype
    x_as_array = np.array(x)
    is_numeric = np.issubdtype(x_as_array.dtype, np.number)

    if is_numeric:
        # only cast numeric data to a real numpy array (due to NaN handling)
        binning = NumericalBinning(x_as_array, nbins)
    else:
        binning = CategoricalBinning(x, nbins)

    if (not insufficient_distinct) and (len(binning) < nbins):
        warnings.warn('Less bins than requested.')

    return binning
