"""Experiment module.
"""

# TODO: question, shouldn't the module level functions be private (_*), because
# the proper interface is through the Experiment instance functions, surely?

# import numpy as np

import re
import expan.core.statistics as statx
import warnings

import expan.core.binning as binmodule  # name conflict with binning...
import numpy as np
import pandas as pd
from expan.core.experimentdata import ExperimentData
from expan.core.results import Results, delta_to_dataframe_all_variants, feature_check_to_dataframe

# raise the same warning multiple times
warnings.simplefilter('always', UserWarning)

from expan.core.debugging import Dbg

def _binned_deltas(df, variants, n_bins=4, binning=None, cumulative=False,
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
	      TODO: currently only the baseline variant is extracted from this list
	        and deltas are calculated for all variants (see bug OCTO-869)

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
		binning = binmodule.create_binning(df.iloc[:, 1], nbins=n_bins)

	if cumulative==True and type(binning)!=binmodule.NumericalBinning:
		raise ValueError("Cannot calculate cumulative deltas for non-numerical binnings")

	# Applying binning to feat1 and feat2 arrays
	df.loc[:, '_tmp_bin_'] = binning.label(data=df.iloc[:, 1],
										   format_str=label_format_str)

	# Initialize result object as data frame with bin keys as index
	def do_delta(f, bin_name):
		# find the corresponding bin in the baseline variant
		baseline_metric = f.iloc[:, 2][(f.iloc[:, 0] == variants[1])]
		out_df=pd.DataFrame()

		for v in f['variant'].unique():
			v_metric =  f.iloc[:, 2][ (f.iloc[:,0]==v) ]
			df = delta_to_dataframe_all_variants(f.columns[2],
													*statx.delta(
														x=v_metric,
														y=baseline_metric,
														assume_normal=assume_normal,
														percentiles=percentiles,
														min_observations=min_observations,
														nruns=nruns, relative=relative))

			# add new index levels for variant and binning
			df['_tmp_bin_']=bin_name
			df['variant'] = v
			df.set_index(['variant', '_tmp_bin_'], append=True, inplace=True)
			df=df.reorder_levels(['variant', '_tmp_bin_', 'metric',
			                      'subgroup_metric', 'subgroup',
			                      'statistic', 'pctile'])

			out_df=out_df.append(df)
		return out_df

	# Actual calculation
	result = pd.DataFrame()
	unique_tmp_bins=df['_tmp_bin_'].unique()
	for bin in unique_tmp_bins:
		if not cumulative:
			result=result.append(do_delta(df[df['_tmp_bin_'] == bin], bin))
		else:
			result=result.append(do_delta(df[df['_tmp_bin_'] <= bin], bin))

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
						nruns=10000, relative=False, weighted=False):
	"""Applies delta to all variants, given a metric and a baseline variant.

	metric_df has 4 columns: entity, variant, metric, reference_kpi
	"""
	baseline_metric = metric_df.iloc[:, 2][metric_df.iloc[:, 1] == baseline_variant]
	baseline_weights = metric_df.iloc[:, 3][metric_df.iloc[:, 1] == baseline_variant]

	if weighted:
		# ASSUMPTIONS:
		# - reference KPI is never NaN (such that sum works the same as np.nansum)
		# - whenever the reference KPI is 0, it means the derived KPI is NaN,
		#	and therefore should not be counted (only works for ratio)
		do_delta = (lambda f: delta_to_dataframe_all_variants(f.columns[2],
															  *statx.delta(
																  x=f.iloc[:, 2],
																  y=baseline_metric,
																  assume_normal=assume_normal,
																  percentiles=percentiles,
																  min_observations=min_observations,
																  nruns=nruns,
																  relative=relative,
																  x_weights=f.iloc[:,3]/sum(f.iloc[:,3])*sum(f.iloc[:,3]!=0),
																  y_weights=baseline_weights/sum(baseline_weights)*sum(baseline_weights!=0))))
	else:
		do_delta = (lambda f: delta_to_dataframe_all_variants(f.columns[2],
															  *statx.delta(
																  x=f.iloc[:, 2],
																  y=baseline_metric,
																  assume_normal=assume_normal,
																  percentiles=percentiles,
																  min_observations=min_observations,
																  nruns=nruns,
																  relative=relative)))
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


def time_dependent_deltas(df, variants, time_step=1, cumulative=False,
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
	#time_bin = (lambda x: round(x / float(time_step) + 0.5) * time_step)

	# Apply time binning vectorized to each element in the input array
	#df['_tmp_time_'] = df.iloc[:, 1].apply(time_bin)

	# Get appropriate bin number
	#n_bins = len(pd.unique(df['_tmp_time_']))

	# create binning manually, ASSUMING uniform sampling
	tpoints = np.unique(df.iloc[:,1])
	binning = binmodule.NumericalBinning(uppers=tpoints, lowers=tpoints,
		up_closed=[True]*len(tpoints), lo_closed=[True]*len(tpoints))

	# Push computation to _binned_deltas() function
	result = _binned_deltas(df=df, variants=variants, binning=binning,
	                        cumulative=cumulative,
							assume_normal=assume_normal, percentiles=percentiles,
							min_observations=min_observations, nruns=nruns,
							relative=relative, label_format_str='{mid}')

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
		# res += '\n KPIs are: \n   {}'.format(
		#		'\n   '.join([('**'+k+'**') if (k == self.metadata.get('primary_KPI','-')) else k for k in self.kpi_names]))

		return res

	def delta(self, kpi_subset=None, derived_kpis=None, variant_subset=None,
			  assume_normal=True, percentiles=[2.5, 97.5],
			  min_observations=20, nruns=10000, relative=False, weighted_kpis=None):
		"""
	    Compute delta (with confidence bounds) on all applicable kpis,
	    and returns in the standard Results format.

	    Does this for all non-baseline variants.

	    TODO: Extend this function to metrics again with type-checking

	    Args:
	        kpi_subset (list): kpis for which to perfom delta. If set to
	            None all kpis are used.
	        derived_kpis (list): definition of additional KPIs derived from the
	        	primary ones, e.g.
	        	[{'name':'return_rate', 'formula':'returned/ordered'}]
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
	        weighted_kpis (list): a list of metric names. For each metric
	        	in the list, the weighted mean and confidence intervals
	        	are calculated, which is equivalent to the overall metric.
	        	Otherwise the metrics are unweighted, this weighted approach
	        	is only relevant for ratios.

	    Returns:
	        Results object containing the computed deltas.
	    """
		res = Results(None, metadata=self.metadata)
		res.metadata['reference_kpi'] = {}
		res.metadata['weighted_kpis'] = weighted_kpis

		# determine the complete KPI name list
		kpis_to_analyse = self.kpi_names.copy()
		if derived_kpis is not None:
			for dk in derived_kpis:
				kpis_to_analyse.update([dk['name']])
				# assuming the columns in the formula can all be cast into float
				# and create the derived KPI as an additional column
				self.kpis.loc[:,dk['name']] = eval(re.sub('([a-zA-Z_]+)', r'self.kpis.\1.astype(float)', dk['formula']))
				# store the reference metric name to be used in the weighting
				# TODO: only works for ratios
				res.metadata['reference_kpi'][dk['name']] = re.sub('([a-zA-Z_]+)/', '', dk['formula'])

		if kpi_subset is not None:
			kpis_to_analyse.intersection_update(kpi_subset)
		self.dbg(3, 'kpis_to_analyse: ' + ','.join(kpis_to_analyse))

		treat_variants = self.variant_names - set([self.baseline_variant])
		self.dbg(3, 'treat_variants before subset: ' + ','.join(treat_variants))
		if variant_subset is not None:
			treat_variants.intersection_update(variant_subset)
		self.dbg(3, 'treat_variants to analyse: ' + ','.join(treat_variants))

		for mname in kpis_to_analyse:
			# the weighted approach implies that derived_kpis is not None
			if weighted_kpis is not None and mname in weighted_kpis:
				reference_kpi = res.metadata['reference_kpi'][mname]
				weighted = True
			else:
				reference_kpi = mname
				weighted = False

			try:
				with warnings.catch_warnings(record=True) as w:
					# Cause all warnings to always be triggered.
					warnings.simplefilter("always")
					df = (_delta_all_variants(self.kpis.reset_index()[['entity', 'variant', mname, reference_kpi]],
											  self.baseline_variant,
											  assume_normal=assume_normal,
											  percentiles=percentiles,
											  min_observations=min_observations,
											  nruns=nruns,
											  relative=relative,
											  weighted=weighted))
					if len(w):
						res.metadata['warnings']['Experiment.delta'] = w[-1].message

					if res.df is None:
						res.df = df
					else:
						res.df = res.df.append(df)

			except ValueError as e:
				res.metadata['errors']['Experiment.delta'] = e

		res.calculate_prob_uplift_over_zero()

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
						self.metrics.reset_index()[['variant', feature, kpi]],
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
			  time_step=1, cumulative=True,
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
		for kpi in kpi_subset:
			for variant in variant_subset:
				# TODO: Add metadata to res.metadata
				res_obj = time_dependent_deltas(
					self.kpis_time.reset_index()[['variant',
												  'time_since_treatment', kpi]],
					variants=[variant, self.baseline_variant],
					time_step=time_step,
					cumulative=cumulative,
					assume_normal=assume_normal,
					percentiles=percentiles,
					min_observations=min_observations,
					nruns=nruns, relative=relative)
				res.df = pd.concat([res.df, res_obj.df])

		# NB: assuming all binning objects based on the same feature are the same
		res.set_binning(res_obj.binning)
		# Return the result object
		return res


if __name__ == '__main__':
	from tests.tests_core.test_data import generate_random_data

	np.random.seed(0)
	metrics, metadata = generate_random_data()
	metrics['time_since_treatment'] = metrics['treatment_start_time']
	exp = Experiment('B', metrics, metadata, [4, 6])
	res = exp.delta(kpi_subset=['derived'],
			derived_kpis=[{'name':'derived','formula':'normal_same/normal_shifted'}],
			weighted_kpis=['derived'])

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
