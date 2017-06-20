import logging
import re
import warnings

import numpy as np
import pandas as pd

import expan.core.binning as binmodule
import expan.core.early_stopping as es
import expan.core.statistics as statx
from expan.core.experimentdata import ExperimentData
from expan.core.results import Results, delta_to_dataframe_all_variants, feature_check_to_dataframe, \
    early_stopping_to_dataframe
from expan.core.util import remove_model_pkls

# raise the same warning multiple times
warnings.simplefilter('always', UserWarning)

logger = logging.getLogger(__name__)

# TODO: add filtering functionality: we should be able to operate on this
# class to exclude data points, and save all these operations in a log that then
# is preserved in all results.
class Experiment(ExperimentData):
    """
    Class which adds the analysis functions to experimental data.
    """
    # TODO: add a constructor that takes an ExperimentData!
    def __init__(self, baseline_variant, metrics_or_kpis, metadata={}, features='default'):
        # Call constructor of super class
        super(Experiment, self).__init__(metrics_or_kpis, metadata, features)

        # If no baseline variant is found
        if ((baseline_variant not in self.kpis.index.levels[
            self.primary_indices.index('variant')])
            and (baseline_variant not in self.features.index.levels[
                self.primary_indices.index('variant')])):
            raise KeyError('baseline_variant ({}) not present in KPIs or features.'.format(
                baseline_variant))
        # Add baseline to metadata
        self.metadata['baseline_variant'] = baseline_variant


    @property
    def baseline_variant(self):
        """
        Returns the baseline variant.

        Returns:
            string: baseline variant
        """
        return self.metadata['baseline_variant']


    def __str__(self):
        res = super(Experiment, self).__str__()

        variants = self.variant_names

        res += '\n {:d} variants: {}'.format(len(variants),
                                             ', '.join(
                                                 [('*' + k + '*') if (k == self.metadata.get('baseline_variant', '-'))
                                                  else k for k in variants]
                                             ))
        return res


    def delta(self, method='fixed_horizon', kpi_subset=None, derived_kpis=None,
              assume_normal=True, percentiles=[2.5, 97.5], min_observations=20,
              nruns=10000, relative=False, weighted_kpis=None):
        """
        Wrapper for different delta functions with 'method' being the following:

        'fixed_horizon': 		self.fixed_horizon_delta()
        'group_sequential': 	self.group_sequential_delta()
        'bayes_factor':			self.bayes_factor_delta()
        'bayes_precision':		self.bayes_precision_delta()
        """
        res = Results(None, metadata=self.metadata)
        res.metadata['reference_kpi'] = {}
        res.metadata['weighted_kpis'] = weighted_kpis

        reference_kpis = {}
        pattern = '([a-zA-Z][0-9a-zA-Z_]*)'

        # determine the complete KPI name list
        kpis_to_analyse = self.kpi_names.copy()
        if derived_kpis is not None:
            for dk in derived_kpis:
                kpiName = dk['name']
                kpis_to_analyse.update([kpiName])
                # assuming the columns in the formula can all be cast into float
                # and create the derived KPI as an additional column
                self.kpis.loc[:, kpiName] = eval(re.sub(pattern, r'self.kpis.\1.astype(float)', dk['formula']))
                # store the reference metric name to be used in the weighting
                # TODO: only works for ratios
                res.metadata['reference_kpi'][kpiName] = re.sub(pattern + '/', '', dk['formula'])
                reference_kpis[kpiName] = re.sub(pattern + '/', '', dk['formula'])

        if kpi_subset is not None:
            kpis_to_analyse.intersection_update(kpi_subset)
        logger.debug('kpis_to_analyse: ' + ','.join(kpis_to_analyse))

        defaultArgs = [res, kpis_to_analyse, reference_kpis, weighted_kpis]
        deltaWorker = statx.make_delta(assume_normal, percentiles, min_observations, nruns, relative)

        alpha = 0.05
        if percentiles is not None and len(percentiles) == 2:
            alpha = (percentiles[0] + 100 - percentiles[1]) / 100

        method_table = {
            'fixed_horizon': (self.fixed_horizon_delta, defaultArgs + [deltaWorker]),
            'group_sequential': (self.group_sequential_delta, defaultArgs + [alpha]),
            'bayes_factor': (self.bayes_factor_delta, defaultArgs),
            'bayes_precision': (self.bayes_precision_delta, defaultArgs),
        }

        if not method in method_table:
            raise NotImplementedError
        else:
            entry = method_table[method]
            f = entry[0]
            vargs = entry[1]
            return f(*vargs)


    def _get_metric_df(self, mname, reference_kpis, weighted_kpis):
        # the weighted approach implies that derived_kpis is not None
        if weighted_kpis is not None and mname in weighted_kpis:
            reference_kpi = reference_kpis[mname]
        else:
            reference_kpi = mname
        metric_df = self.kpis.reset_index()[['entity', 'variant', mname, reference_kpi]]
        return metric_df


    def _apply_reweighting_and_all_variants(self,
                                            df_grouped_by_variant,
                                            metric_df,
                                            weighted_kpis,
                                            reference_kpis,
                                            mname,
                                            func_apply_variants):
        """
        Compute the reweighting trick to derived KPIs,
        and apply regular delta or early stopping method to all variants.

        metric_df has 4 columns: entity, variant, metric, reference_kpi
        """
        ctrl_kpis = metric_df.iloc[:, 2][metric_df.iloc[:, 1] == self.baseline_variant]
        treat_kpis = df_grouped_by_variant.iloc[:, 2]

        # the weighted approach implies that derived_kpis is not None
        if weighted_kpis is not None and mname in weighted_kpis:
            reference_kpi = reference_kpis[mname]

            ctrl_reference_kpis = metric_df.loc[metric_df.variant == self.baseline_variant, reference_kpi]
            treat_reference_kpis = df_grouped_by_variant.loc[:, reference_kpi]

            n_nan_ref_ctrl = sum(ctrl_reference_kpis == 0) + np.isnan(ctrl_reference_kpis).sum()
            n_non_nan_ref_ctrl = len(ctrl_reference_kpis) - n_nan_ref_ctrl

            n_nan_ref_treat = sum(treat_reference_kpis == 0) + np.isnan(treat_reference_kpis).sum()
            n_non_nan_ref_treat = len(treat_reference_kpis) - n_nan_ref_treat

            ctrl_weights = n_non_nan_ref_ctrl * ctrl_reference_kpis / np.nansum(ctrl_reference_kpis)
            treat_weights = n_non_nan_ref_treat * treat_reference_kpis / np.nansum(treat_reference_kpis)

        else:
            ctrl_weights = 1.
            treat_weights = 1.

        return func_apply_variants(x=treat_kpis, y=ctrl_kpis, x_weights=treat_weights, y_weights=ctrl_weights)


    def fixed_horizon_delta(self,
                            res,
                            kpis_to_analyse=None,
                            reference_kpis={},
                            weighted_kpis=None,
                            deltaWorker=statx.make_delta(),
                            **kwargs):
        """
        Compute delta (with confidence bounds) on all applicable kpis,
        and returns in the standard Results format.

        Does this for all non-baseline variants.

        TODO: Extend this function to metrics again with type-checking

        Args:
            res: a Results object which is to be extended.
            kpis_to_analyse (list): kpis for which to perfom delta. If set to
                None all kpis are used.
            reference_kpis: dict with the names of reference KPIs parameterized
                by the name of the primary KPI
            weighted_kpis: names of KPIs that should be treated as weighted
            deltaWorker: a closure over statistics.delta() as described by
                statistics.make_delta()

        Returns:
            Results object containing the computed deltas.
        """
        kpis_to_analyse = kpis_to_analyse or self.kpi_names.copy()

        for mname in kpis_to_analyse:
            metric_df = self._get_metric_df(mname, reference_kpis, weighted_kpis)
            def do_delta(x, y, x_weights, y_weights):
                return delta_to_dataframe_all_variants(metric_df.columns[2],
                                                       *deltaWorker(x=x,
                                                                    y=y,
                                                                    x_weights=x_weights,
                                                                    y_weights=y_weights))
            try:
                with warnings.catch_warnings(record=True) as w:
                    # Cause all warnings to always be triggered.
                    warnings.simplefilter("always")
                    df = metric_df.groupby('variant').apply(self._apply_reweighting_and_all_variants,
                                                            metric_df=metric_df,
                                                            weighted_kpis=weighted_kpis,
                                                            reference_kpis=reference_kpis,
                                                            mname=mname,
                                                            func_apply_variants=do_delta).unstack(0)
                    if len(w):
                        res.metadata['warnings']['Experiment.delta'] = w[-1].message

                    if res.df is None:
                        res.df = df
                    else:
                        res.df = res.df.append(df)

            except ValueError as e:
                res.metadata['errors']['Experiment.delta'] = e

        return res


    def group_sequential_delta(self,
                               result,
                               kpis_to_analyse,
                               reference_kpis={},
                               weighted_kpis=None,
                               alpha=0.05,
                               spending_function='obrien_fleming',
                               information_fraction=1,
                               cap=8,
                               **kwargs):
        """
        Calculate the stopping criterion based on the group sequential design 
        and the effect size.

        Args:
            result: the initialized Results object
            kpis_to_analyse: list of KPIs to be analysed
            spending_function: name of the alpha spending function, currently
                supports: 'obrien_fleming'
            information_fraction: share of the information amount at the point 
                of evaluation, e.g. the share of the maximum sample size
            alpha: type-I error rate
            cap: upper bound of the adapted z-score

        Returns:
            a Results object
        """
        if 'estimatedSampleSize' not in self.metadata:
            raise ValueError("Missing 'estimatedSampleSize' for the group sequential method!")

        for mname in kpis_to_analyse:
            metric_df = self._get_metric_df(mname, reference_kpis, weighted_kpis)

            current_sample_size = float(sum(~metric_df.iloc[:,2].isnull()))
            information_fraction = current_sample_size / self.metadata['estimatedSampleSize']

            def do_delta(x, y, x_weights, y_weights):
                weighted_x = np.array(x, dtype=float) * x_weights
                weighted_y = np.array(y, dtype=float) * y_weights
                return early_stopping_to_dataframe(metric_df.columns[2],
                                                   *es.group_sequential(
                                                       x=weighted_x,
                                                       y=weighted_y,
                                                       spending_function=spending_function,
                                                       information_fraction=information_fraction,
                                                       alpha=alpha,
                                                       cap=cap))

            df = metric_df.groupby('variant').apply(self._apply_reweighting_and_all_variants,
                                                    metric_df=metric_df,
                                                    weighted_kpis=weighted_kpis,
                                                    reference_kpis=reference_kpis,
                                                    mname=mname,
                                                    func_apply_variants=do_delta).unstack(0)
            # force the stop label of the baseline variant to 0
            df.loc[(mname, '-', slice(None), 'stop'), ('value', self.baseline_variant)] = 0

            if result.df is None:
                result.df = df
            else:
                result.df = result.df.append(df)

        return result


    def bayes_factor_delta(self,
                           result,
                           kpis_to_analyse,
                           reference_kpis={},
                           weighted_kpis=None,
                           distribution='normal',
                           num_iters=25000,
                           remove_pkls=True,
                           **kwargs):
        """
        Calculate the stopping criterion based on the Bayes factor 
        and the effect size.

        Args:
            result: the initialized Results object
            kpis_to_analyse: list of KPIs to be analysed
            distribution: name of the KPI distribution model, which assumes a
                Stan model file with the same name exists
            num_iters: number of iterations for the bayes sampling

        Returns:
            a Results object
        """
        for mname in kpis_to_analyse:
            metric_df = self._get_metric_df(mname, reference_kpis, weighted_kpis)

            def do_delta(x, y, x_weights, y_weights):
                weighted_x = np.array(x, dtype=float) * x_weights
                weighted_y = np.array(y, dtype=float) * y_weights
                return early_stopping_to_dataframe(metric_df.columns[2],
                                                   *es.bayes_factor(
                                                       x=weighted_x,
                                                       y=weighted_y,
                                                       distribution=distribution,
                                                       num_iters=num_iters))

            df = metric_df.groupby('variant').apply(self._apply_reweighting_and_all_variants,
                                                    metric_df=metric_df,
                                                    weighted_kpis=weighted_kpis,
                                                    reference_kpis=reference_kpis,
                                                    mname=mname,
                                                    func_apply_variants=do_delta).unstack(0)
            # force the stop label of the baseline variant to 0
            df.loc[(mname, '-', slice(None), 'stop'), ('value', self.baseline_variant)] = 0

            if result.df is None:
                result.df = df
            else:
                result.df = result.df.append(df)

        # Remove .pkl compiled stan model files
        if remove_pkls:
            remove_model_pkls()

        return result


    def bayes_precision_delta(self,
                              result,
                              kpis_to_analyse,
                              reference_kpis={},
                              weighted_kpis=None,
                              distribution='normal',
                              posterior_width=0.08,
                              num_iters=25000,
                              remove_pkls=True,
                              **kwargs):
        """
        Calculate the stopping criterion based on the precision of the posterior 
        and the effect size.

        Args:
            result: the initialized Results object
            kpis_to_analyse: list of KPIs to be analysed
            distribution: name of the KPI distribution model, which assumes a
                Stan model file with the same name exists
            posterior_width: the stopping criterion, threshold of the posterior 
                width
            num_iters: number of iterations for the bayes sampling

        Returns:
            a Results object
        """
        for mname in kpis_to_analyse:
            metric_df = self._get_metric_df(mname, reference_kpis, weighted_kpis)

            def do_delta(x, y, x_weights, y_weights):
                weighted_x = np.array(x, dtype=float) * x_weights
                weighted_y = np.array(y, dtype=float) * y_weights
                return early_stopping_to_dataframe(metric_df.columns[2],
                                                   *es.bayes_precision(
                                                       x=weighted_x,
                                                       y=weighted_y,
                                                       distribution=distribution,
                                                       posterior_width=posterior_width,
                                                       num_iters=num_iters))

            df = metric_df.groupby('variant').apply(self._apply_reweighting_and_all_variants,
                                                    metric_df=metric_df,
                                                    weighted_kpis=weighted_kpis,
                                                    reference_kpis=reference_kpis,
                                                    mname=mname,
                                                    func_apply_variants=do_delta).unstack(0)
            # force the stop label of the baseline variant to 0
            df.loc[(mname, '-', slice(None), 'stop'), ('value', self.baseline_variant)] = 0

            if result.df is None:
                result.df = df
            else:
                result.df = result.df.append(df)

        # Remove .pkl compiled stan model files
        if remove_pkls:
            remove_model_pkls()

        return result


    def feature_check(self, feature_subset=None, variant_subset=None,
                      threshold=0.05, percentiles=[2.5, 97.5], assume_normal=True,
                      min_observations=20, nruns=10000, relative=False):

        """
        Compute feature check on all features, and return dataframe with column
        telling if feature check passed.

        Args:
            feature_subset (list): Features for which to perfom delta. If set to
                None all metrics are used.
            variant_subset (list): Variants to use compare against baseline. If
                set to None all variants are used.
            threshold (float): p-value used for dismissing null hypothesis (i.e.
                no difference between features for variant and baseline).

            assume_normal (boolean): specifies whether normal distribution
                assumptions can be made
            min_observations (integer): minimum observations necessary. If
                less observations are given, then NaN is returned
            nruns (integer): number of bootstrap runs to perform if assume
                normal is set to False.

        Returns:
            pd.DataFrame containing boolean column named 'ok' stating if
                feature chek was ok for the feature and variant combination
                specified in the corresponding columns.
        """
        # TODO: this should return a results structure, like all the others?
        # - can monkey patch it with a function to just get the 'ok' column

        res = Results(None, metadata=self.metadata)

        # Check if data exists TODO: Necessary or guarantted by __init__() ?
        if self.features is None:
            warnings.warn('Empty data set entered to analysis.'
                          + 'Returning empty result set')
            return res
        # TODO: Check if subsets are valid
        # If no subsets use superset
        if feature_subset is None:
            feature_subset = self.feature_names
        if variant_subset is None:
            variant_subset = self.variant_names

        deltaWorker = statx.make_delta(assume_normal, percentiles, min_observations,
                                       nruns, relative)
        # Iterate over the features
        for feature in feature_subset:
            df = (self._feature_check_all_variants(self.features.reset_index()[['entity', 'variant', feature]],
                                                   self.baseline_variant, deltaWorker))
            if res.df is None:
                res.df = df
            else:
                res.df = res.df.append(df)

        return res


    def sga(self, feature_subset=None, kpi_subset=None, variant_subset=None,
            n_bins=4, binning=None,
            assume_normal=True, percentiles=[2.5, 97.5],
            min_observations=20, nruns=10000, relative=False,
            **kwargs):
        """
        Compute subgroup delta (with confidence bounds) on all applicable
        metrics, and returns in the standard Results format.

        Does this for all non-baseline variants.

        Args:
            feature_subset (list): Features which are binned for which to
                perfom delta computations. If set to None all features are used.
            kpi_subset (list): KPIs for which to perfom delta computations.
                If set to None all features are used.
            variant_subset (list): Variants to use compare against baseline. If
                set to None all variants are used.
            n_bins (integer): number of bins to create if binning is None

            binning (list of bins): preset (if None then binning is created)
            assume_normal (boolean): specifies whether normal distribution
                assumptions can be made
            percentiles (list): list of percentile values to compute
            min_observations (integer): minimum observations necessary. If
                less observations are given, then NaN is returned
            nruns (integer): number of bootstrap runs to perform if assume
                normal is set to False.
            relative (boolean): If relative==True, then the values will be
                returned as distances below and above the mean, respectively,
                rather than the	absolute values. In	this case, the interval is
                mean-ret_val[0] to mean+ret_val[1]. This is more useful in many
                situations because it corresponds with the sem() and std()
                functions.

        Returns:
            Results object containing the computed deltas.
        """
        res = Results(None, metadata=self.metadata)

        # Check if data exists
        if self.metrics is None:
            warnings.warn('Empty data set entered to analysis.'
                          + 'Returning empty result set')
            return res
        # TODO: Check if subsets are valid
        # If no subsets use superset
        if kpi_subset is None:
            kpi_subset = self.kpi_names
        if feature_subset is None:
            feature_subset = self.feature_names
        if variant_subset is None:
            variant_subset = self.variant_names
        # Remove baseline from variant_set
        variant_subset = variant_subset - set([self.baseline_variant])
        # Iterate over the kpis, features and variants
        # TODO: Check if this is the right approach,
        # groupby and unstack as an alternative?
        deltaWorker = statx.make_delta(assume_normal, percentiles, min_observations,
                                       nruns, relative)
        for kpi in kpi_subset:
            for feature in feature_subset:
                res.df = pd.concat([
                    res.df,
                    self._subgroup_deltas(
                        self.metrics.reset_index()[['variant', feature, kpi]],
                        variants=['dummy', self.baseline_variant],
                        n_bins=n_bins,
                        deltaWorker=deltaWorker).df])
        # Return the result object
        return res


    def trend(self, kpi_subset=None, variant_subset=None, time_step=1,
              cumulative=True, assume_normal=True, percentiles=[2.5, 97.5],
              min_observations=20, nruns=10000, relative=False, **kwargs):
        """
        Compute time delta (with confidence bounds) on all applicable
        metrics, and returns in the standard Results format.

        Does this for all non-baseline variants.

        Args:
            kpi_subset (list): KPIs for which to perfom delta computations.
                If set to None all features are used.
            variant_subset (list): Variants to use compare against baseline. If
                set to None all variants are used.
            time_step (integer): time increment over which to aggregate data.
            cumulative (boolean): Trend is calculated using data from
                start till the current bin or the current bin only

            assume_normal (boolean): specifies whether normal distribution
                assumptions can be made
            percentiles (list): list of percentile values to compute
            min_observations (integer): minimum observations necessary. If
                less observations are given, then NaN is returned
            nruns (integer): number of bootstrap runs to perform if assume
                normal is set to False.
            relative (boolean): If relative==True, then the values will be
                returned as distances below and above the mean, respectively,
                rather than the	absolute values. In	this case, the interval is
                mean-ret_val[0] to mean+ret_val[1]. This is more useful in many
                situations because it corresponds with the sem() and std()
                functions.

        Returns:
            Results object containing the computed deltas.
        """
        res = Results(None, metadata=self.metadata)
        # Check if data exists
        if self.kpis_time is None:
            warnings.warn('Empty data set entered to analysis. '
                          + 'Returning empty result set')
            res.metadata['warnings']['Experiment.trend'] = \
                UserWarning('Empty data set entered to analysis.')
            return res
        # Check if time is in dataframe column
        if 'time_since_treatment' not in self.kpis_time.index.names:
            warnings.warn('Need time column for trend analysis.'
                          + 'Returning empty result set')
            res.metadata['warnings']['Experiment.trend'] = \
                UserWarning('Need time column for trend analysis.')
            return res
        # TODO: Check if subsets are valid
        # If no subsets use superset
        if kpi_subset is None:
            kpi_subset = self.kpi_names
        if variant_subset is None:
            variant_subset = self.variant_names
        # Remove baseline from variant_set
        variant_subset = variant_subset - set([self.baseline_variant])
        # Iterate over the kpis and variants
        # TODO: Check if this is the right approach
        deltaWorker = statx.make_delta(assume_normal, percentiles, min_observations,
                                       nruns, relative)
        for kpi in kpi_subset:
            for variant in variant_subset:
                # TODO: Add metadata to res.metadata
                res_obj = self._time_dependent_deltas(
                    self.kpis_time.reset_index()[['variant',
                                                  'time_since_treatment', kpi]],
                    variants=[variant, self.baseline_variant],
                    time_step=time_step,
                    cumulative=cumulative,
                    deltaWorker=deltaWorker)
                res.df = pd.concat([res.df, res_obj.df])

        # NB: assuming all binning objects based on the same feature are the same
        res.set_binning(res_obj.binning)
        # Return the result object
        return res


    def _feature_check_all_variants(self, metric_df, baseline_variant, deltaWorker):
        """Applies delta to all variants, given a metric."""
        baseline_metric = metric_df.iloc[:, 2][metric_df.variant == baseline_variant]

        def do_delta_numerical(df):
            mu, ci, ss_x, ss_y, mean_x, mean_y = deltaWorker(x=df.iloc[:, 2],
                                                             y=baseline_metric)
            return feature_check_to_dataframe(metric=df.columns[2],
                                              samplesize_variant=ss_x,
                                              mu=mu,
                                              pctiles=ci,
                                              mu_variant=mean_x)

        def do_delta_categorical(df):
            pval = statx.chi_square(x=df.iloc[:, 2], y=baseline_metric)[0]
            ss_x = statx.sample_size(df.iloc[:, 2])
            return feature_check_to_dataframe(metric=df.columns[2],
                                              samplesize_variant=ss_x,
                                              pval=pval)

        # numerical feature
        if np.issubdtype(metric_df.iloc[:, 2].dtype, np.number):
            return metric_df.groupby('variant').apply(do_delta_numerical).unstack(0)
        # categorical feature
        else:
            return metric_df.groupby('variant').apply(do_delta_categorical).unstack(0)


    def _time_dependent_deltas(self, df, variants, time_step=1, cumulative=False,
                               deltaWorker=statx.make_delta()):
        """
        Calculates the time dependent delta.

        Args:
          df (pandas DataFrame): 3 columns. The order of the columns is expected
              to be variant, time, kpi.
          variants (list of 2): 2 entries, first entry is the treatment variant,
              second entry specifies the baseline variant
          time_step (integer): time_step used for analysis.
          cumulative (Boolean): whether to accumulate values over time
          deltaWorker: a closure generated by statitics.make_delta(), holding
              the numerical parameters of delta calculations

        Returns:
          pandas.DataFrame: bin-name, mean, percentile and corresponding values
          list: binning used
        """
        # TODO: allow times to have time stamp format
        # TODO: allow start time and end time format
        # TODO: fill with zeros

        # Create time binning with time_step
        # time_bin = (lambda x: round(x / float(time_step) + 0.5) * time_step)

        # Apply time binning vectorized to each element in the input array
        # df['_tmp_time_'] = df.iloc[:, 1].apply(time_bin)

        # Get appropriate bin number
        # n_bins = len(pd.unique(df['_tmp_time_']))

        # create binning manually, ASSUMING uniform sampling
        tpoints = np.unique(df.iloc[:, 1])
        binning = binmodule.NumericalBinning(uppers=tpoints, lowers=tpoints,
                                             up_closed=[True] * len(tpoints), lo_closed=[True] * len(tpoints))

        # Push computation to _binned_deltas() function
        result = self._binned_deltas(df=df, variants=variants, binning=binning,
                                     cumulative=cumulative, label_format_str='{mid}',
                                     deltaWorker=deltaWorker)

        # Reformating of the index names in the result data frame object
        result.df.index.set_names('time', level=2, inplace=True)

        # Returning Result object containing result and the binning
        return result


    def _subgroup_deltas(self, df, variants, n_bins=4, deltaWorker=statx.make_delta()):
        """
        Calculates the feature dependent delta.
    
        Args:
          df (pandas DataFrame): 3 columns. The order of the columns is expected
              to be variant, feature, kpi.
          variants (list of 2): 2 entries, first entry is the treatment variant,
              second entry specifies the baseline variant
          n_bins (integer): number of bins to create if binning is None
          deltaWorker: a closure generated by statitics.make_delta(), holding
              the numerical parameters of delta calculations
    
    
        Returns:
          pandas.DataFrame: bin-name, mean, percentile and corresponding values
          list: binning used
        """

        # Push computation to _binned_deltas() function
        result = self._binned_deltas(df=df, variants=variants, n_bins=n_bins, binning=None,
                                     cumulative=False, label_format_str='{standard}',
                                     deltaWorker=deltaWorker)

        # TODO: Add binning to result metadata

        # Reformating of the index names in the result data frame object
        result.df.reset_index('subgroup', drop=True, inplace=True)
        result.df.index.set_names('subgroup', level=2, inplace=True)
        result.df.index.set_levels(levels=[df.columns[1]],
                                   level='subgroup_metric', inplace=True)

        # Returning Result object containing result and the binning
        return result


    def _binned_deltas(self, df, variants, n_bins=4, binning=None, cumulative=False,
                       label_format_str='{standard}', deltaWorker=statx.make_delta()):
        """
        Calculates the feature dependent delta. Only used internally. All
        calculation by subgroup_delta() and time_dependant_delta() is pushed here.
    
        Args:
          df (pandas DataFrame): 3 columns. The order of the columns is expected
              to be variant, feature, kpi.
          variants (list of 2): 2 entries, first entry is the treatment variant,
              second entry specifies the baseline variant
              TODO: currently only the baseline variant is extracted from this list
              and deltas are calculated for all variants (see bug OCTO-869)
          n_bins (integer): number of bins to create if binning is None
          binning (list of bins): preset (if None then binning is created)
          cumulative (Bool): whether to accumulate data (for time trend analysis)
          label_format_str (string): format string for the binning label function
          deltaWorker: a closure generated by statitics.make_delta(), holding
              the numerical parameters of delta calculations
    
        Returns:
          pandas.DataFrame: bin-name, mean, percentile and corresponding values
          list: binning used
        """

        # Performing binning of feature on feat2
        if binning is None:
            binning = binmodule.create_binning(df.iloc[:, 1], nbins=n_bins)

        if cumulative == True and type(binning) != binmodule.NumericalBinning:
            raise ValueError("Cannot calculate cumulative deltas for non-numerical binnings")

        # Applying binning to feat1 and feat2 arrays
        df.loc[:, '_tmp_bin_'] = binning.label(data=df.iloc[:, 1],
                                               format_str=label_format_str)

        # Initialize result object as data frame with bin keys as index
        def do_delta(f, bin_name):
            # find the corresponding bin in the baseline variant
            baseline_metric = f.iloc[:, 2][(f.iloc[:, 0] == variants[1])]
            out_df = pd.DataFrame()

            for v in f['variant'].unique():
                v_metric = f.iloc[:, 2][(f.iloc[:, 0] == v)]
                df = delta_to_dataframe_all_variants(f.columns[2], *deltaWorker(x=v_metric,
                                                                                y=baseline_metric))
                # add new index levels for variant and binning
                df['_tmp_bin_'] = bin_name
                df['variant'] = v
                df.set_index(['variant', '_tmp_bin_'], append=True, inplace=True)
                df = df.reorder_levels(['variant', '_tmp_bin_', 'metric',
                                        'subgroup_metric', 'subgroup',
                                        'statistic', 'pctile'])

                out_df = out_df.append(df)
            return out_df

        # Actual calculation
        result = pd.DataFrame()
        unique_tmp_bins = df['_tmp_bin_'].unique()
        for bin in unique_tmp_bins:
            if not cumulative:
                result = result.append(do_delta(df[df['_tmp_bin_'] == bin], bin))
            else:
                result = result.append(do_delta(df[df['_tmp_bin_'] <= bin], bin))

        # unstack variant
        result = result.unstack(0)
        # drop _tmp_bin_ in the input data frame
        del df['_tmp_bin_']

        result.index = result.index.swaplevel(0, 2)
        result.index = result.index.swaplevel(0, 1)
        # Return result and binning
        return Results(result, {'binning': binning})
