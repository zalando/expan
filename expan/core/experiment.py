import logging
import re
import warnings

import numpy as np
import pandas as pd

import expan.core.early_stopping as es
import expan.core.statistics as statx
from expan.core.util import get_column_names_by_type
from expan.core.version import __version__
from expan.core.binning import create_bins

warnings.simplefilter('always', UserWarning)

logger = logging.getLogger(__name__)


# TODO: add filtering functionality: we should be able to operate on this
# class to exclude data points, and save all these operations in a log that then
# is preserved in all results.
class Experiment(object):
    """
    Class which adds the analysis functions to experimental data.
    """
    def __init__(self, control_variant_name, data, metadata, report_kpi_names=None, derived_kpis=None):
        report_kpi_names = report_kpi_names or []
        derived_kpis = derived_kpis or []

        experiment_column_names = set(['entity', 'variant'])
        numerical_column_names = set(get_column_names_by_type(data, np.number))

        if type(report_kpi_names) is str:
            report_kpi_names = [report_kpi_names]

        if type(report_kpi_names) is not list:
            raise TypeError('report_kpi_names should be a list of str')

        if report_kpi_names:
            report_kpi_names_needed = set(report_kpi_names)
        else:
            report_kpi_names_needed = numerical_column_names - experiment_column_names

        # check derived_kpis structure (should have keys namely 'name' and 'formula')
        for i in derived_kpis:
            if not isinstance(i, dict):
                raise TypeError('Derived kpis should be an array of dictionaries')
            if 'formula' not in i:
                raise KeyError('Dictionary should have key "formula"')
            if 'name' not in i:
                raise KeyError('Dictionary should have key "name"')

        derived_kpi_names    = [k['name']    for k in derived_kpis]
        derived_kpi_formulas = [k['formula'] for k in derived_kpis]

        # what columns do we expect to find in the data frame?
        required_column_names = (report_kpi_names_needed | experiment_column_names) - set(derived_kpi_names)
        kpi_name_pattern = '([a-zA-Z][0-9a-zA-Z_]*)'
        # add names from all formulas
        for formula in derived_kpi_formulas:
            names = re.findall(kpi_name_pattern, formula)
            required_column_names = required_column_names | set(names)

        for c in required_column_names:
            if c not in data:
                raise ValueError('No column %s provided'%c)

        self.data                 =     data.copy()
        self.metadata             = metadata.copy()
        self.report_kpi_names     = report_kpi_names_needed
        self.derived_kpis         = derived_kpis
        self.variant_names        = set(self.data.variant)
        self.control_variant_name = control_variant_name
        self.reference_kpis       = {}

        # add derived KPIs to the data frame
        for name, formula in zip(derived_kpi_names, derived_kpi_formulas):
            self.data.loc[:, name] = eval(re.sub(kpi_name_pattern, r'self.data.\1.astype(float)', formula))
            self.reference_kpis[name] = re.sub(kpi_name_pattern + '/', '', formula)

    def get_kpi_by_name_and_variant(self, data, name, variant):
        return data.reset_index().set_index('variant').loc[variant, name]

    def __str__(self):
        variants = self.variant_names

        return 'Experiment "{:s}" with {:d} derived kpis, {:d} report kpis, {:d} entities and {:d} variants: {}'.format(
            self.metadata['experiment'], len(self.derived_kpis), len(self.report_kpi_names), len(self.data),
            len(variants), ', '.join([('*' + k + '*') if (k == self.control_variant_name) else k for k in variants]))

    def _get_weights(self, data, kpi, variant):
        if kpi not in self.reference_kpis:
            return 1.0
        reference_kpi  = self.reference_kpis[kpi]
        x              = self.get_kpi_by_name_and_variant(data, reference_kpi, variant)
        zeros_and_nans = sum(x == 0) + np.isnan(x).sum()
        non_zeros      = len(x) - zeros_and_nans
        return non_zeros/np.nansum(x) * x

    def delta(self, method='fixed_horizon', **worker_args):
        return self._delta(method=method, data=self.data, **worker_args)

    def _delta(self, method, data, **worker_args):
        # entity should be unique
        if data.entity.duplicated().any():
            raise ValueError('Entities in data should be unique')

        worker_table = {
            'fixed_horizon'    : statx.make_delta,
            'group_sequential' : es.make_group_sequential,
            'bayes_factor'     : es.make_bayes_factor,
            'bayes_precision'  : es.make_bayes_precision
        }

        if not method in worker_table:
            raise NotImplementedError

        worker = worker_table[method](**worker_args)

        result = {'warnings': [],
                  'errors': [],
                  'expan_version': __version__,
                  'control_variant': self.control_variant_name}
        kpis = []

        for kpi in self.report_kpi_names:
            res_kpi = {'name': kpi,
                       'variants': []}
            control         = self.get_kpi_by_name_and_variant(data, kpi, self.control_variant_name)
            control_weight  = self._get_weights(data, kpi, self.control_variant_name)
            control_data    = control * control_weight
            for variant in self.variant_names:
                treatment        = self.get_kpi_by_name_and_variant(data, kpi, variant)
                treatment_weight = self._get_weights(data, kpi, variant)
                treatment_data   = treatment * treatment_weight
                with warnings.catch_warnings(record=True) as w:
                    statistics = worker(x=treatment_data, y=control_data)
                    # add statistical power
                    power = statx.compute_statistical_power(treatment_data, control_data)
                    statistics['statistical_power'] = power
                if len(w):
                    result['warnings'].append('kpi: {}, variant: {}: {}'.format(kpi, variant, w[-1].message))
                res_kpi['variants'].append({'name': variant, 'delta_statistics': statistics})
            kpis.append(res_kpi)

        result['kpis'] = kpis
        return result

    def _quantile_filtering(self, kpis, percentile, threshold_type):
        method_table = {'upper': lambda x: x > threshold, 'lower': lambda x: x <= threshold}
        flags = pd.Series(data=[False]*len(self.data))
        for column in self.data[kpis].columns:
            threshold = np.percentile(self.data[column], percentile)
            flags = flags | self.data[column].apply(method_table[threshold_type])
        return flags

    def filter(self, kpis, percentile=99.0, threshold_type='upper'):
        """
        Method that filters out entities whose KPIs exceed the value at a given percentile.
        If any of the KPIs exceeds its threshold the entity is filtered out.

        Args:
            kpis (list): list of KPI names
            percentile (float): percentile considered as threshold
            threshold_type (string): type of threshold used ('lower' or 'upper')

        Returns:
            No return value. Will filter out outliers in self.data in place.
        """

        # check if provided KPIs are present in the data
        for kpi in kpis:
            if kpi not in self.data.columns:
                raise KeyError(kpi + ' identifier not present in dataframe columns!')

        # check if provided percentile is valid
        if 0.0 < percentile <= 100.0 is False:
            raise ValueError("Percentile value needs to be between 0.0 and 100.0!")

        # check if provided filtering kind is valid
        if threshold_type not in ['upper', 'lower']:
            raise ValueError("Threshold type needs to be either 'upper' or 'lower'!")

        # run quantile filtering
        flags = self._quantile_filtering(kpis=kpis, percentile=percentile, threshold_type=threshold_type)

        # log which columns were filtered and how many entities were filtered out
        self.metadata['filtered_columns'] = kpis
        self.metadata['filtered_entities_number'] = len(flags[flags == True])
        self.metadata['filtered_threshold_kind'] = threshold_type

        # throw warning if too many entities have been filtered out
        if (len(flags[flags == True]) / float(len(self.data))) > 0.02:
            warnings.warn('More than 2% of entities have been filtered out, consider adjusting the percentile value.')

        self.data = self.data[flags == False]

    def sga(self, feature_name_to_bins):
        """
        Perform subgroup analysis.
        Args:
            feature_name_to_bins (dict): a dict of feature name (key) to list of Bin objects (value). 
                                      This dict defines how and on which column to perform the subgroup split.
        Returns:
            Analysis results per subgroup. 
        """

        for feature in feature_name_to_bins:
            # check type
            if type(feature) is not str:
                raise TypeError("Key of the input dict needs to be string, indicating the name of dimension.")
            if type(feature_name_to_bins[feature]) is not list:
                raise TypeError("Value of the input dict needs to be a list of Bin objects.")
            # check whether data contains this column
            if feature not in self.data:
                raise KeyError('No column %s provided in data.' % feature)

        subgroups = []
        for feature in feature_name_to_bins:
            for bin in feature_name_to_bins[feature]:
                subgroup = {'dimension': feature,
                            'segment': str(bin.representation)}
                subgroup_data = bin(self.data, feature)

                if subgroup_data is None:
                    continue

                subgroup_res = self._delta(method='fixed_horizon', data=subgroup_data,
                                           num_tests=len(self.report_kpi_names))
                subgroup['result'] = subgroup_res
                subgroups.append(subgroup)

        return subgroups

    def sga_date(self):
        """
        Perform subgroup analysis on date partitioning each day from start day till end date. Produces non-cumulative
        delta and CIs for each subgroup.
        Returns:
            Analysis results per date
        """

        if 'date' not in self.data:
            raise KeyError('No column date provided in data.')

        num_bins = len(set(self.data['date']))
        bins = create_bins(self.data['date'], num_bins)

        subgroups = []
        for bin in bins:
            subgroup = {'dimension': 'date',
                        'segment': str(bin.representation)}
            subgroup_data = bin(self.data, 'date')
            subgroup_res = self._delta(method='fixed_horizon', data=subgroup_data,
                                       num_tests=len(self.report_kpi_names))
            subgroup['result'] = subgroup_res
            subgroups.append(subgroup)

        return subgroups
