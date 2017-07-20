import logging
import re
import warnings

import numpy as np
import pandas as pd

import expan.core.binning as binmodule
import expan.core.early_stopping as es
import expan.core.statistics as statx

from expan.core.util import getColumnNamesByType
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
    def __init__(self, controlVariantName, data, metadata, reportKpiNames=None, derivedKpis=[]):
        experimentColumnNames = set(['entity', 'variant'])
        numericalColumnNames  = set(getColumnNamesByType(data, np.number))

        if reportKpiNames:
            reportKpiNames = set(reportKpiNames)
        else:
            reportKpiNames =  numericalColumnNames - experimentColumnNames

        derivedKpiNames    = [k['name']    for k in derivedKpis]
        derivedKpiFormulas = [k['formula'] for k in derivedKpis]

        # what columns do we expect to find in the data frame?
        requiredColumnNames = (reportKpiNames | experimentColumnNames) - set(derivedKpiNames)
        kpiNamePattern = '([a-zA-Z][0-9a-zA-Z_]*)'
        # add names from all formulas
        for formula in derivedKpiFormulas:
            names = re.findall(kpiNamePattern, formula)
            requiredColumnNames = requiredColumnNames | set(names)

        for c in requiredColumnNames:
            if c not in data:
                raise ValueError('No column %s provided'%c)

        self.data               =     data.copy()
        self.metadata           = metadata.copy()
        self.reportKpiNames     = reportKpiNames
        self.derivedKpis        = derivedKpis
        self.variantNames       = set(self.data.variant)
        self.controlVariantName = controlVariantName
        self.referenceKpis      = {}

        # add derived KPIs to the data frame
        for name, formula in zip(derivedKpiNames, derivedKpiFormulas):
            self.data.loc[:, name] = eval(re.sub(kpiNamePattern, r'self.data.\1.astype(float)', formula))
            self.referenceKpis[name] = re.sub(kpiNamePattern + '/', '', formula)



    def getKPIbyNameAndVariant(self, name, variant):
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



    def _getWeights(self, kpi, variant):
        if kpi not in self.referenceKpis:
            return 1.0
        referenceKpi = self.referenceKpis[kpi]
        x            = self.getKPIbyNameAndVariant(referenceKpi, variant)
        zerosAndNans = sum(x == 0) + np.isnan(x).sum()
        nonZeros     = len(x) - zerosAndNans
        return nonZeros/np.nansum(x) * x

    def delta(self, method='fixed_horizon', **workerArgs):
        workerTable = {
                'fixed_horizon'    : statx.make_delta,
                'group_sequential' : es.make_group_sequential,
                'bayes_factor'     : es.make_bayes_factor,
                'bayes_precision'  : es.make_bayes_precision,
                }

        if not method in workerTable:
            raise NotImplementedError

        worker = workerTable[method](**workerArgs)

        result = {}
        for kpi in self.reportKpiNames:
            result[kpi] = {}
            control       = self.getKPIbyNameAndVariant(kpi, self.controlVariantName)
            controlWeight = self._getWeights(kpi, self.controlVariantName)
            for variant in self.variantNames:
                treatment       = self.getKPIbyNameAndVariant(kpi, variant)
                treatmentWeight = self._getWeights(kpi, variant)
                ds = worker(x=treatment*treatmentWeight, y=control*controlWeight)
                result[kpi][variant] = {'controlVariant'   : self.controlVariantName,
                                        'treatmentVariant' : variant,
                                        'deltaStatistics'  : ds}

        return result
