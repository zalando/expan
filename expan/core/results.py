from __future__ import absolute_import
import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from expan.core.version import __version__
import expan.core.statistics as statx
from scipy.stats import norm

# from tests.tests_core.test_data import generate_random_data

from expan.core.debugging import Dbg

class Results(object):
	"""
	A Results instance represents the results of a series of analyses such as
	SGA or deltaKPI.

	Q: could we make this a subclass of DataFrame (i.e. provide this class in an
	'is-a' relationship with DataFrame, rather than the 'has-a' relationship it
	has now? It seems to be a difficult thing to subclass the DataFrame object
	for some reason. https://github.com/pydata/pandas/pull/4271
	For now, will leave as a 'has-a' class.

	Todo:
		Can we remove the 'value' level from the columns, so that the columns of the dataframe are simply the names of the variants?
		This will make the columns a normal index rather than a multi-index. Currently, always a multi-index with second level only containing a single value 'value'.
	"""

	#TODO: maybe move these two to the __init__?
	mandatory_index_levels = [
		'metric',
		'subgroup_metric',
		'subgroup',
		'statistic',
		'pctile']
	mandatory_column_levels = ['variant']

	def __init__(self, df, metadata={}, dbg=None):
		"""
	    Want to be able to create results from just a single dataframe.

	    Args:
	        df (pandas.DataFrame): input dataframe
	        metadata (dict): input metadata
	        dbg:
	    """
		self.df = df
		self.metadata = metadata
		self.metadata['version'] = __version__
		self.metadata['errors'] = {}
		self.metadata['warnings'] = {}

		self.dbg = dbg or Dbg()

	@property
	def binning(self):
		"""Return the binning object."""
		return self.metadata['binning']

	def set_binning(self, binning):
		"""Store a binning object in the metadata."""
		self.metadata['binning'] = binning

	def _sortlevels(self):
		self.df.sortlevel(axis=0, inplace=True, sort_remaining=True)
		self.df.sortlevel(axis=1, inplace=True, sort_remaining=True)

	def append_delta(self, metric, variant, mu, pctiles,
					 samplesize_variant,
					 samplesize_baseline,
					 subgroup_metric='-',
					 subgroup=None):
		"""
	    Appends the results of a delta.

	    Modifies (or creates) the results data (df).

	    Args:
	        metric:
	        variant:
	        mu:
	        pctiles:
	        samplesize_variant:
	        samplesize_baseline:
	        subgroup_metric:
	        subgroup:
	    """
		df = delta_to_dataframe(metric, variant, mu, pctiles,
								samplesize_variant,
								samplesize_baseline,
								subgroup_metric='-',
								subgroup=None)

		if self.df is None:
			self.df = df
		else:
			self.df = self.df.append(df)

		self._sortlevels()

	def variants(self):
		"""
	    Return the variants represented in this object
	    """
		return self.df.columns.levels[0]

	def index_values(self, level='metric'):
		"""
	    Return the metrics represented in this Results object
	    """
		return self.df.index.get_level_values(level=level).unique()

	def relative_uplift(self, analysis_type, metric=None, subgroup_metric='-'):
		"""Calculate the relative uplift for the given metrics and subgroup
	    metrics.
	    """
		uplifts = self.statistic(analysis_type, 'uplift', metric, subgroup_metric)
		baseline_abs = self.statistic(analysis_type, 'variant_mean', metric,
									  subgroup_metric).loc[:, ('value', self.metadata['baseline_variant'])]
		val = uplifts.values / baseline_abs.values[:, np.newaxis]

		df = pd.DataFrame(val, columns=uplifts.columns)
		# reconstruct indices
		for i in self.mandatory_index_levels:
			df[i] = uplifts.index.get_level_values(i)
		df.set_index(self.mandatory_index_levels, inplace=True)
		# df.index.set_levels(['uplift_rel']*df.index.shape[0], level='statistic', inplace=True)

		# TODO: do we return a data frame or a Results object here?
		return df

	def calculate_prob_uplift_over_zero(self):
		"""

		"""
		# check if the subgroup index is NaN
		# NB: this will NOT work if we store delta and SGA results in the same object
		# if the subgroup index contains only NaNs
		if len(self.df.index.levels[2]) == 0:
			df = self.df.groupby(level=['metric']).apply(lambda x:prob_uplift_over_zero_single_metric(x,self.metadata['baseline_variant']))
			# remove redundant levels (coming from groupby)
			df.reset_index(level=0,drop=True,inplace=True)
		else:
			df = self.df.groupby(level=['metric','subgroup_metric','subgroup']).apply(lambda x:prob_uplift_over_zero_single_metric(x,self.metadata['baseline_variant']))
			# remove redundant levels (coming from groupby)
			df.reset_index(level=[0,1,2],drop=True,inplace=True)

		self.df = df
		#return df

	def delta_means(self, metric=None, subgroup_metric='-'):
		"""

		Args:
		    metric:
		    subgroup_metric:

		Returns:

		"""
		return self.statistic('delta', 'variant_mean', metric, subgroup_metric)

	def sga_means(self, metric=None, subgroup_metric='-'):
		"""

		Args:
		    metric:
		    subgroup_metric:

		Returns:

		"""
		return self.statistic('sga', 'variant_mean', metric, subgroup_metric)

	def uplifts(self, metric=None, subgroup_metric='-'):
		"""

		Args:
		    metric:
		    subgroup_metric:

		Returns:

		"""
		return self.statistic('delta', 'uplift', metric, subgroup_metric)

	def sga_uplifts(self, metric=None, subgroup_metric='-'):
		"""

		Args:
		    metric:
		    subgroup_metric:

		Returns:

		"""
		return self.statistic('sga', 'uplift', metric, subgroup_metric)

	def sample_sizes(self, analysis_type='delta', metric=None, subgroup_metric='-'):
		"""

		Args:
		    analysis_type:
		    metric:
		    subgroup_metric:

		Returns:

		"""
		return self.statistic(analysis_type, 'sample_size', metric, subgroup_metric)

	def statistic(self, analysis_type, statistic=None, metric=None,
				  subgroup_metric='-',
				  time_since_treatment='-',
				  include_pctiles=True):
		"""
	    This is just a basic 'formatter' to allow easy access to results without
	    knowing the ordering of the index, etc. and to have sensible defaults.
	    All of this can be accomplished with fancy indexing on the dataframe
	    directly, but this should just serve as a convenience and an obvious
	    place to 'document' what the typical use-case is.

	    For all arguments, None means all, and '-' means only those for which
	    this particular argument is undefined (e.g. subgroup_metric='-')

	    Args:

	            analysis_type (string): the type of analysis that produced the  TODO: implement this!
	                results (several can be present in a single Result object). Must be
	                one of the following:
	                        - 'delta':	only those with no time_since_treatment information, and no subgroup defined
	                        - 'sga':		only those with subgroup defined
	                        - 'trend':	only those with time_since_treatment defined
	                        - None:			no restriction done

	            statistic (string): the type of data you want, such as 'uplift'

	            metric (string): which metrics you are interested in

	            time_since_treatment (int?): TODO: implement

	            include_pctiles (bool): some statistics (e.g. 'uplift') can be present
	                with percentiles defined, as opposed to just a mean. If this is true,
	                they'll be returned also. TODO: implement this!
	    """
		# NOTE: throws AssertionError for trend results and unittest example results
		# assert (self.df.index.names[0:4] == Results.mandatory_index_levels[0:4])
		# assert (self.df.columns.names[1] == Results.mandatory_column_levels[0])

		mean_results = self.df.xs((statistic, metric, subgroup_metric),
								  level=('statistic', 'metric', 'subgroup_metric'), drop_level=False)

		# metric_slicer = metric or slice(None)
		# statistic_slicer = stat or slice(None)

		# rows = (metric,subgroup_metric,slice(None),statistic_slicer)

		# levels_to_drop = ['subgroup_metric', 'statistic']
		# if subgroup_metric == '-':
		# 	levels_to_drop.append('subgroup')
		# if metric is not None:
		# 	levels_to_drop.insert(0, 'metric')

		# if stat == 'pctile':
		# 	cols = (slice(None))
		# else:
		# 	cols = (slice(None),'value')
		# 	levels_to_drop.append('pctile')

		# mean_results = self.df.loc[rows, cols]

		# #mean_results.index = mean_results.reset_index(levels_to_drop,drop=True)
		# mean_results = mean_results.reset_index(levels_to_drop,drop=True)
		# mean_results.columns = mean_results.columns.droplevel(1)


		return mean_results

	def bounds(self, metric=None, subgroup_metric='-'):
		"""

		Args:
		    metric:
		    subgroup_metric:

		Returns:

		"""
		if False:
			rows = (slice(None), '-', slice(None), ['uplift', 'uplift_pctile'])
			cols = (slice(None), 'value')
			results = self.df.loc[rows, cols].unstack(['statistic', 'pctile'])

			results.columns = results.columns.droplevel(1)

			if subgroup_metric == '-':
				results.reset_index(['subgroup', 'subgroup_metric'], drop=True)
		else:
			pctiles = self.statistic('pctile').unstack('pctile')
			mns = self.uplifts()
			mns.columns = pd.MultiIndex.from_product(
				(self.means().columns, ['mean']), names=['variant', 'statistic'])
			results = pd.concat((pctiles, mns), axis=1)
			results.columns.names = ['variant', 'statistic']

		return results

	def __str__(self):
		# TODO: improve
		return 'Results for \'{}\' with {:d} variants, {:d} metrics, {:d} subgroup metrics. Means are:\n{}'.format(
			str(self.metadata.get('experiment')),
			len(self.variants()),
			len(self.index_values('metric')),
			len(self.index_values('subgroup_metric')),
			str(self.means()),
		)

	def __repr__(self):
		return 'Results(metadata={}, \ndf={})'.format(repr(self.metadata),
													  repr(self.df.unstack('pctile')))

	def to_csv(self, fpath):
		"""

		Args:
		    fpath:

		Returns:

		Note:
		    This will lose all metadata.
		"""

		res = deepcopy(self.df)
		res.columns = res.columns.droplevel(0)
		res = res.reset_index()
		res.to_csv(fpath, index=False)

	def to_hdf(self, fpath):
		"""
	    Persist to an HDF5 file, preserving metadata.

	    MetaData is stored as attributes on a Group called 'metadata'. This group
	    doesn't include any datasets, but was used to avoid interfering with the
	    attributes that pandas stores on the 'data' Group.

		Args:
		    fpath:

		Returns:

	    """
		import h5py

		store = pd.HDFStore(fpath)
		store['data'] = self.df  # write to HDF5
		store.close()

		# TODO: surely can do this via single interface rather than opening and closing?
		hfile = h5py.File(fpath)
		md = hfile.require_group('metadata')
		datetime_conversions = set(md.attrs.get('_datetime_attributes', set()))
		for k, v in list(self.metadata.items()):
			if k == '_datetime_attributes':
				continue
			if v is None:
				continue
			self.dbg(3, 'to_hdf: storing metadata {}'.format(k))
			if isinstance(v, pd.Timestamp) or isinstance(v, datetime.datetime):
				v = str(v)
				self.dbg(3, ' -> converted datetime/timestamp to string')
				datetime_conversions.add(k)
			md.attrs[k] = v

		if len(datetime_conversions) > 0:
			md.attrs['_datetime_attributes'] = [str(x) for x in datetime_conversions]

		hfile.close()


def prob_uplift_over_zero_single_metric(result_df, baseline_variant):
	"""Calculate the probability of uplift>0 for a single metric.

	Args:
		result_df (DataFrame): result data frame of a single metric/subgroup
		baseline_variant (str): name of the baseline variant

	Returns:
		DataFrame: result data frame with one additional statistic 'prob_uplift_over_0'
	"""
	pctile = 97.5 # result should be independent of the percentile that we choose
	all_variants = set(result_df.columns.levels[1])
	# iterate over all non-baseline variants
	variant = all_variants - set([baseline_variant])
	#set_trace()
	prob_dict = {baseline_variant:np.nan}
	for v in variant:
		mu = float(result_df.xs(('uplift'),level=('statistic'))[('value',v)])
		x = float(result_df.xs(('uplift_pctile',pctile),level=('statistic','pctile'))[('value',v)])
		sigma = statx.estimate_std(x, mu, pctile)

		prob = 1 - norm.cdf(0, loc=mu, scale=sigma)
		prob_dict[v] = prob

	# convert dict to df
	prob_list = []
	for v in result_df.columns.levels[1]:
		prob_list.append(prob_dict[v])
	# a prob df w/o multi-index
	prob_df = pd.DataFrame([prob_list], columns=result_df.columns)
	# reconstruct indices
	for i in ['metric','subgroup_metric','subgroup']:
		prob_df[i] = result_df.index.get_level_values(i)[0]
	prob_df['statistic'] = 'prob_uplift_over_0'
	prob_df['pctile'] = np.nan
	prob_df.set_index(Results.mandatory_index_levels, inplace=True)

	ret = pd.concat((result_df,prob_df))
	ret.sort_index(inplace=True)

	return ret

def from_hdf(fpath, dbg=None):
	"""
	Restores a Results object from HDF5 as created by the to_hdf method.

	Args:
	    fpath:
	    dbg:

	Returns:

	"""
	if dbg is None:
		dbg = Dbg()

	import h5py

	data = pd.read_hdf(fpath, 'data')

	hfile = h5py.File(fpath)
	md = hfile['metadata']
	datetime_conversions = set(md.attrs.get('_datetime_attributes', set()))
	metadata = {}
	for k, v in list(md.attrs.items()):
		if k == '_datetime_attributes':
			continue
		dbg(3, 'from_hdf: retrieving metadata {}'.format(k))
		if k in datetime_conversions:
			dbg(3, ' -> converting to Timestamp')
			v = pd.Timestamp(v)
		metadata[k] = v

	return Results(data, metadata)


def delta_to_dataframe(metric, variant, mu, pctiles, samplesize_variant, samplesize_baseline,
					   subgroup_metric='-',
					   subgroup=None):
	"""Defines the Results data frame structure.

	Args:
	    metric:
	    variant:
	    mu:
	    pctiles:
	    samplesize_variant:
	    samplesize_baseline:
	    subgroup_metric:
	    subgroup:

	Returns:

	"""

	df = pd.DataFrame({
		'metric': metric,
		'variant': variant,
		'statistic': 'pctile',
		'pctile': list(pctiles.keys()),
		'value': list(pctiles.values()),
		'subgroup_metric': subgroup_metric,
		'subgroup': subgroup
	})
	# TODO: put baseline in as separate column... no need for sample_size_baseline
	df = df.append(pd.DataFrame({
		'metric': metric,
		'variant': variant,
		'statistic': ['mean', 'sample_size', 'sample_size_baseline'],
		'value': [mu, samplesize_variant, samplesize_baseline],
		'subgroup_metric': subgroup_metric,
		'subgroup': subgroup
	}), ignore_index=True)

	df.set_index(Results.mandatory_index_levels + ['variant'], inplace=True)
	df = df.unstack('variant')
	df.columns = df.columns.swaplevel(0, 1)

	return df


def delta_to_dataframe_all_variants(metric, mu, pctiles, samplesize_variant,
									samplesize_baseline, mu_variant,
									mu_baseline,
									subgroup_metric='-',
									subgroup=None):
	"""Defines the Results data frame structure.

	Args:
	    metric:
	    mu:
	    pctiles:
	    samplesize_variant:
	    samplesize_baseline:
	    mu_variant:
	    mu_baseline:
	    subgroup_metric:
	    subgroup:

	Returns:

	"""

	df = pd.DataFrame({
		'metric': metric,
		'statistic': 'uplift_pctile',
		'pctile': list(pctiles.keys()),
		'value': list(pctiles.values()),
		'subgroup_metric': subgroup_metric,
		'subgroup': subgroup
	})
	df = df.append(pd.DataFrame({
		'metric': metric,
		'statistic': ['uplift', 'sample_size', 'variant_mean'],
		'value': [mu, samplesize_variant, mu_variant],
		'subgroup_metric': subgroup_metric,
		'subgroup': subgroup
	}), ignore_index=True)

	df.set_index(Results.mandatory_index_levels, inplace=True)
	# df = df.unstack('variant')
	# df.columns = df.columns.swaplevel(0,1)

	return df


def feature_check_to_dataframe(metric,
							   samplesize_variant,
							   mu=None,
							   pctiles=None,
							   pval=None,
							   mu_variant=None):
	"""Defines the Results data frame structure.

	Args:
	    metric:
	    samplesize_variant:
	    mu:
	    pctiles:
	    pval:
	    mu_variant:

	Returns:

	"""

	# numerical feature
	if pval is None:
		df = pd.DataFrame({'metric': metric,
						   'statistic': 'pre_treatment_diff_pctile',
						   'pctile': list(pctiles.keys()),
						   'value': list(pctiles.values()),
						   'subgroup_metric': '-',
						   'subgroup': None})
		df = df.append(pd.DataFrame({
			'metric': metric,
			'statistic': ['pre_treatment_diff', 'sample_size', 'variant_mean'],
			'value': [mu, samplesize_variant, mu_variant],
			'subgroup_metric': '-',
			'subgroup': None
		}), ignore_index=True)
	# categorical feature
	if mu is None:
		df = pd.DataFrame({
			'metric': metric,
			'pctile': None,
			'statistic': ['chi_square_p', 'sample_size'],
			'value': [pval, samplesize_variant],
			'subgroup_metric': '-',
			'subgroup': None
		})

	df.set_index(Results.mandatory_index_levels, inplace=True)
	# df = df.unstack('variant')
	# df.columns = df.columns.swaplevel(0,1)

	return df


if __name__ == '__main__':
	#pass

	np.random.seed(0)
	from tests.tests_core.test_data import generate_random_data
	from expan.core.experiment import Experiment
	data = Experiment('B', *generate_random_data())
	res = data.delta(kpi_subset=['normal_same','normal_shifted'])
	# df = res.calculate_prob_uplift_over_zero()

	# from test_core.test_results import load_example_results
	# aa = load_example_results()
	# order_means = aa.means('orders').iloc[0]
	# net_sales_var = aa.statistic('var', 'net_sales')

	# import numpy as np
	# res = Results(None)
	# res.append_delta('dummy', 'A', *(0.1,{'2.5':0.01,'97.5':0.2},1000,1000))
	# res.append_delta('dummy', 'B', *(0,{'2.5':np.nan,'97.5':np.nan},1000,1000))

	# from expan.core.experiment import Experiment
    #
	# np.random.seed(0)
	# data = Experiment('B', *generate_random_data())
	# res = data.sga()
	# x = res.relative_uplift('sga', 'normal_same', 'feature')
# res = data.delta()
# x = res.relative_uplift('delta', 'normal_same')
