import logging
import re
import warnings

import numpy as np
import pandas as pd

import expan.core.binning as binmodule
import expan.core.early_stopping as es
import expan.core.statistics as statx

from expan.core.util import get_column_names_by_type
# from expan.core.experimentdata import ExperimentData
# from expan.core.results import Results, delta_to_dataframe_all_variants, feature_check_to_dataframe, \
#     early_stopping_to_dataframe

# from expan.core.jsonable import Jsonable

# raise the same warning multiple times
warnings.simplefilter('always', UserWarning)

logger = logging.getLogger(__name__)

# TODO: add filtering functionality: we should be able to operate on this
# class to exclude data points, and save all these operations in a log that then
# is preserved in all results.
class Experiment(object):
    """
    Class which adds the analysis functions to experimental data.
    """
    def __init__(self, control_variant_name, data, metadata, report_kpi_names=None, derived_kpis=[]):
        experiment_column_names = set(['entity', 'variant'])
        numerical_column_names  = set(get_column_names_by_type(data, np.number))

        if report_kpi_names:
            report_kpi_names = set(report_kpi_names)
        else:
            report_kpi_names =  numerical_column_names - experiment_column_names

        derived_kpi_names    = [k['name']    for k in derived_kpis]
        derived_kpi_formulas = [k['formula'] for k in derived_kpis]

        # what columns do we expect to find in the data frame?
        required_column_names = (report_kpi_names | experiment_column_names) - set(derived_kpi_names)
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
        self.report_kpi_names     = report_kpi_names
        self.derived_kpis         = derived_kpis
        self.variant_names        = set(self.data.variant)
        self.control_variant_name = control_variant_name
        self.reference_kpis       = {}

        # add derived KPIs to the data frame
        for name, formula in zip(derived_kpi_names, derived_kpi_formulas):
            self.data.loc[:, name] = eval(re.sub(kpi_name_pattern, r'self.data.\1.astype(float)', formula))
            self.reference_kpis[name] = re.sub(kpi_name_pattern + '/', '', formula)



    def get_kpi_by_name_and_variant(self, name, variant):
        return self.data.reset_index().set_index('variant').loc[variant, name]


    def __str__(self):
        # res = super(Experiment, self).__str__()

        variants = self.variant_names

        res += '\n {:d} variants: {}'.format(len(variants),
                                             ', '.join(
                                                 [('*' + k + '*') if (k == self.metadata.get('baseline_variant', '-'))
                                                  else k for k in variants]
                                             ))
        return res



    def _get_weights(self, kpi, variant):
        if kpi not in self.reference_kpis:
            return 1.0
        reference_kpi  = self.reference_kpis[kpi]
        x              = self.get_kpi_by_name_and_variant(reference_kpi, variant)
        zeros_and_nans = sum(x == 0) + np.isnan(x).sum()
        non_zeros      = len(x) - zeros_and_nans
        return non_zeros/np.nansum(x) * x

    def delta(self, method='fixed_horizon', **workerArgs):
        worker_table = {
                'fixed_horizon'    : statx.make_delta,
                'group_sequential' : es.make_group_sequential,
                'bayes_factor'     : es.make_bayes_factor,
                'bayes_precision'  : es.make_bayes_precision,
                }

        if not method in worker_table:
            raise NotImplementedError

        worker = worker_table[method](**workerArgs)

        result = {}
        for kpi in self.report_kpi_names:
            result[kpi] = {}
            control       = self.get_kpi_by_name_and_variant(kpi, self.control_variant_name)
            controlWeight = self._get_weights(kpi, self.control_variant_name)
            for variant in self.variant_names:
                treatment       = self.get_kpi_by_name_and_variant(kpi, variant)
                treatmentWeight = self._get_weights(kpi, variant)
                ds = worker(x=treatment*treatmentWeight, y=control*controlWeight)
                result[kpi][variant] = {'control_variant'   : self.control_variant_name,
                                        'treatment_variant' : variant,
                                        'delta_statistics'  : ds}

        return result
