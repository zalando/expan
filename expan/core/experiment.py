# TODO: question, shouldnt the module level functions be private (_*), because
# the proper interface is through the Experiment instance functions, surely?

# import numpy as np
import statistics as statx
import warnings

import binning as binmodule  # name conflict with binning...
import numpy as np
import pandas as pd
from experimentdata import ExperimentData
from results import Results, delta_to_dataframe_all_variants, feature_check_to_dataframe

# raise the same warning multiple times
warnings.simplefilter('always', UserWarning)

from debugging import Dbg


def _binned_deltas(df, variants, n_bins=4, binning=None,
				   assume_normal=True, percentiles=[2.5, 97.5],
				   min_observations=20, nruns=10000, relative=False,
				   label_format_str='{standard}'):
	"""
  Calculates the feature dependent delta. Only used internally. All
  calculation by subgroup_delta() and time_dependant_delta() is pushed here.

  Args:
      df (pandas DataFrame): 3 columns. The order of the columns is expected
          to be variant, feature, kpi.
      variants (list of 2): 2 entries, first entry is the treatment variant,
          second entry specifies the baseline variant

      n_bins (integer): number of bins to create if binning is None
      binning (list of bins): preset (if None then binning is created)

      percentiles (list): list of percentile values to compute
      min_observations (integer): minimum number of observations necessary. If
          less observations are given, then NaN is returned.
      nruns (integer): number of bootstrap runs to perform if assume normal is
          set to False.
      relative (boolean): If relative==True, then the values will be returned
          as distances below and above the mean, respectively, rather than the
          absolute values. In	this case, the interval is mean-ret_val[0] to
          mean+ret_val[1]. This is more useful in many situations because it
          corresponds with the sem() and std() functions.
      label_format_str (string): format string for the binning label function.

  Returns:
      pandas.DataFrame: bin-name, mean, percentile and corresponding values
      list: binning used
  """

	# Performing binning of feature on feat2
	if binning is None:
		cg_feat_df = df.iloc[:, 1][df.iloc[:, 0] == variants[1]]
		binning = binmodule.create_binning(df.iloc[:, 1], nbins=n_bins)

	# Applying binning to feat1 and feat2 arrays
	df.loc[:, '_tmp_bin_'] = binning.label(data=df.iloc[:, 1],
										   format_str=label_format_str)

	# Initialize result object as data frame with bin keys as index
	def do_delta(f):
		# find the corresponding bin in the baseline variant
		baseline_metric = df.iloc[:, 2][
			(df.iloc[:, 0] == variants[1]) & (df['_tmp_bin_'] == f['_tmp_bin_'].tolist()[0])]
		return (delta_to_dataframe_all_variants(f.columns[2],
												*statx.delta(
													x=f.iloc[:, 2],
													y=baseline_metric,
													assume_normal=assume_normal,
													percentiles=percentiles,
													min_observations=min_observations,
													nruns=nruns, relative=relative)))

	# Actual calculation
	result = df.groupby(['variant', '_tmp_bin_']).apply(do_delta)
	# unstack variant
	result = result.unstack(0)
	# drop _tmp_bin_ in the input data frame
	del df['_tmp_bin_']

	result.index = result.index.swaplevel(0, 2)
	result.index = result.index.swaplevel(0, 1)
	# Return result and binning
	return Results(result, {'binning': binning})


def _delta_all_variants(metric_df, baseline_variant, assume_normal=True,
						percentiles=[2.5, 97.5], min_observations=20,
						nruns=10000, relative=False):
	"""Applies delta to all variants, given a metric."""
	baseline_metric = metric_df.iloc[:, 2][metric_df.iloc[:, 1] == baseline_variant]
	do_delta = (lambda f: delta_to_dataframe_all_variants(f.columns[2],
														  *statx.delta(
															  x=f.iloc[:, 2],
															  y=baseline_metric,
															  assume_normal=assume_normal,
															  percentiles=percentiles,
															  min_observations=min_observations,
															  nruns=nruns, relative=relative)))
	# Actual calculation
	return metric_df.groupby('variant').apply(do_delta).unstack(0)


def _feature_check_all_variants(metric_df, baseline_variant, assume_normal=True,
								percentiles=[2.5, 97.5], min_observations=20,
								nruns=10000, relative=False):
	"""Applies delta to all variants, given a metric."""
	baseline_metric = metric_df.iloc[:, 2][metric_df.variant == baseline_variant]

	def do_delta_numerical(df):
		mu, ci, ss_x, ss_y, mean_x, mean_y = statx.delta(x=df.iloc[:, 2],
														 y=baseline_metric,
														 assume_normal=assume_normal,
														 percentiles=percentiles,
														 min_observations=min_observations,
														 nruns=nruns,
														 relative=relative)
		return feature_check_to_dataframe(metric=df.columns[2],
										  samplesize_variant=ss_x,
										  mu=mu,
										  pctiles=ci,
										  mu_variant=mean_x)

	def do_delta_categorical(df):
		pval = statx.chi_square(x=df.iloc[:, 2], y=baseline_metric)
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


def subgroup_deltas(df, variants, n_bins=4, binning=None,
					assume_normal=True, percentiles=[2.5, 97.5],
					min_observations=20, nruns=10000, relative=False):
	"""
  Calculates the feature dependent delta.

  Args:
      df (pandas DataFrame): 3 columns. The order of the columns is expected
          to be variant, feature, kpi.
      variants (list of 2): 2 entries, first entry is the treatment variant,
          second entry specifies the baseline variant

      n_bins (integer): number of bins to create if binning is None
      binning (list of bins): preset (if None then binning is created)

      assume_normal (boolean): specifies whether normal distribution
          assumptions can be made
      percentiles (list): list of percentile values to compute
      min_observations (integer): minimum number of observations necessary. If
          less observations are given, then NaN is returned.
      nruns (integer): number of bootstrap runs to perform if assume normal is
          set to False.
      relative (boolean): If relative==True, then the values will be returned
          as distances below and above the mean, respectively, rather than the
          absolute values. In	this case, the interval is mean-ret_val[0] to
          mean+ret_val[1]. This is more useful in many situations because it
          corresponds with the sem() and std() functions.

  Returns:
      pandas.DataFrame: bin-name, mean, percentile and corresponding values
      list: binning used
  """

	# Push computation to _binned_deltas() function
	result = _binned_deltas(df=df, variants=variants, n_bins=n_bins,
							assume_normal=assume_normal, percentiles=percentiles,
							min_observations=min_observations, nruns=nruns,
							relative=relative)

	# TODO: Add binning to result metadata

	# Reformating of the index names in the result data frame object
	result.df.reset_index('subgroup', drop=True, inplace=True)
	result.df.index.set_names('subgroup', level=2, inplace=True)
	result.df.index.set_levels(levels=[df.columns[1]],
							   level='subgroup_metric', inplace=True)

	# Returning Result object containing result and the binning
	return result


def time_dependent_deltas(df, variants, time_step=1,
						  assume_normal=True, percentiles=[2.5, 97.5],
						  min_observations=20, nruns=10000, relative=False):
	"""
  Calculates the time dependent delta.

  Args:
      df (pandas DataFrame): 3 columns. The order of the columns is expected
          to be variant, time, kpi.
      variants (list of 2): 2 entries, first entry is the treatment variant,
          second entry specifies the baseline variant

      time_step (integer): time_step used for analysis.

      percentiles (list): list of percentile values to compute
      min_observations (integer): minimum number of observations necessary. If
          less observations are given, then NaN is returned
      nruns (integer): number of bootstrap runs to perform if assume normal is
          set to False.
      relative (boolean): If relative==True, then the values will be returned
          as distances below and above the mean, respectively, rather than the
          absolute values. In	this case, the interval is mean-ret_val[0] to
          mean+ret_val[1]. This is more useful in many situations because it
          corresponds with the sem() and std() functions.

  Returns:
      pandas.DataFrame: bin-name, mean, percentile and corresponding values
      list: binning used
  """
	# TODO: allow times to have time stamp format
	# TODO: allow start time and end time format
	# TODO: fill with zeros

	# Create time binning with time_step
	time_bin = (lambda x: round(x / float(time_step) + 0.5) * time_step)

	# Apply time binning vectorized to each element in the input array
	df['_tmp_time_'] = df.iloc[:, 1].apply(time_bin)

	# Get appropriate bin number
	n_bins = len(pd.unique(df['_tmp_time_']))

	# Push computation to _binned_deltas() function
	result = _binned_deltas(df=df, variants=variants, n_bins=n_bins,
							assume_normal=assume_normal, percentiles=percentiles,
							min_observations=min_observations, nruns=nruns,
							relative=relative, label_format_str=None)

	# Reformating of the index names in the result data frame object
	result.df.index.set_names('time', level=2, inplace=True)

	# Returning Result object containing result and the binning
	return result


################################################################################


# TODO: can we unbury this class a bit - be nice to have it as the first
# thing in the file?
# TODO: add filtering functionality: we should be able to operate on this
# class to exclude data points, and save all these operations in a log that then
# is preserved in all results.
class Experiment(ExperimentData):
	"""
  Class which adds the analysis functions to experimental data.
  """

	# TODO: rearrange arguments
	# TODO: add a constructor that takes an ExperimentData!
	def __init__(self, baseline_variant, metrics_or_kpis, metadata={},
				 features='default', dbg=None):
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

		self.dbg = dbg or Dbg()

	@property
	def baseline_variant(self):
		return self.metadata['baseline_variant']

	def __str__(self):
		res = super(Experiment, self).__str__()

		variants = self.variant_names

		res += '\n {:d} variants: {}'.format(len(variants),
											 ', '.join(
												 [('*' + k + '*') if (k == self.metadata.get('baseline_variant', '-'))
												  else k for k in variants]
											 ))
		# res += '\n KPIs are: \n   {}'.format(
		#		'\n   '.join([('**'+k+'**') if (k == self.metadata.get('primary_KPI','-')) else k for k in self.kpi_names]))

		return res

	def delta(self, kpi_subset=None, variant_subset=None,
			  assume_normal=True, percentiles=[2.5, 97.5],
			  min_observations=20, nruns=10000, relative=False):
		"""
    Compute delta (with confidence bounds) on all applicable kpis,
    and returns in the standard Results format.

    Does this for all non-baseline variants.

    TODO: Extend this function to metrics again with type-checking

    Args:
        kpi_subset (list): kpis for which to perfom delta. If set to
            None all kpis are used.
        variant_subset (list): Variants to use compare against baseline. If
            set to None all variants are used.

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

		kpis_to_analyse = self.kpi_names.copy()
		if kpi_subset is not None:
			kpis_to_analyse.intersection_update(kpi_subset)
		self.dbg(3, 'kpis_to_analyse: ' + ','.join(kpis_to_analyse))

		treat_variants = self.variant_names - set([self.baseline_variant])
		self.dbg(3, 'treat_variants before subset: ' + ','.join(treat_variants))
		if variant_subset is not None:
			treat_variants.intersection_update(variant_subset)
		self.dbg(3, 'treat_variants to analyse: ' + ','.join(treat_variants))

		for mname in kpis_to_analyse:
			try:
				with warnings.catch_warnings(record=True) as w:
					# Cause all warnings to always be triggered.
					warnings.simplefilter("always")
					df = (_delta_all_variants(self.kpis.reset_index()[['entity', 'variant', mname]],
											  self.baseline_variant, assume_normal=assume_normal,
											  percentiles=percentiles, min_observations=min_observations,
											  nruns=nruns, relative=relative))
					if len(w):
						res.metadata['warnings']['Experiment.delta'] = w[-1].message

					if res.df is None:
						res.df = df
					else:
						res.df = res.df.append(df)

			except ValueError as e:
				res.metadata['errors']['Experiment.delta'] = e

		return res

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

		# Iterate over the features
		for feature in feature_subset:
			df = (_feature_check_all_variants(self.features.reset_index()[['entity', 'variant', feature]],
											  self.baseline_variant, assume_normal=assume_normal,
											  percentiles=percentiles, min_observations=min_observations,
											  nruns=nruns, relative=relative))
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
		for kpi in kpi_subset:
			for feature in feature_subset:
				res.df = pd.concat([
					res.df,
					subgroup_deltas(
						self.metrics.reset_index() \
							[['variant', feature, kpi]],
						variants=['dummy', self.baseline_variant],
						n_bins=n_bins,
						binning=binning,
						assume_normal=assume_normal,
						percentiles=percentiles,
						min_observations=min_observations,
						nruns=nruns, relative=relative).df])
		# Return the result object
		return res

	def trend(self, kpi_subset=None, variant_subset=None,
			  time_step=1,
			  assume_normal=True, percentiles=[2.5, 97.5],
			  min_observations=20, nruns=10000, relative=False,
			  **kwargs):
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
			warnings.warn('Empty data set entered to analysis.'
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
		for kpi in kpi_subset:
			for variant in variant_subset:
				# TODO: Add metadata to res.metadata
				res_obj = time_dependent_deltas(
					self.kpis_time.reset_index()[['variant',
												  'time_since_treatment', kpi]],
					variants=[variant, self.baseline_variant],
					time_step=time_step,
					assume_normal=assume_normal,
					percentiles=percentiles,
					min_observations=min_observations,
					nruns=nruns, relative=relative)
				res.df = pd.concat([res.df, res_obj.df])

		# NB: assuming all binning objects based on the same feature are the same
		res.set_binning(res_obj.binning)
		# Return the result object
		return res


# if __name__ == '__main__':
# 	from tests.tests_core.test_data import generate_random_data
#
# 	np.random.seed(0)
# 	data = Experiment('B', *generate_random_data())
# 	# Create time column. TODO: Do this nicer
# 	# data.kpis['time_since_treatment'] = data.features['treatment_start_time']
# 	# Make time part of index
# 	# data.kpis.set_index('time_since_treatment',append=True,inplace=True)
#
# 	data.kpis['time_since_treatment'] = data.features['treatment_start_time']
# 	data.kpis.set_index('time_since_treatment', append=True, inplace=True)
# 	result = data.delta(kpi_subset=['normal_unequal_variance'],
# 						variant_subset=['A'])

# result = time_dependent_deltas(data.metrics.reset_index()
#	[['variant','time_since_treatment','normal_shifted']],variants=['A','B']).df.loc[:,1]
# result = result.reset_index('subgroup',drop=True)
# result['subgroup'] = np.nan

# result = data.feature_check()
# res = _delta_all_variants(data.kpis.reset_index()[['entity','variant','normal_shifted']], 'A')
# result = data.feature_check()

# result = time_dependent_deltas(data.metrics.reset_index()
#				[['variant','treatment_start_time','normal_shifted']],variants=['A','B'])
# df = result.statistic('delta', 'uplift', 'normal_shifted', 'feature')
# result = data.delta(kpi_subset=['normal_unequal_variance'], variant_subset=['A'])
