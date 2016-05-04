"""
ExperimentData module allows us to abstract the data format from the library
such that we can have multiple fetcher modules that only need import this
module, and none of our analysis modules need know anything about the fetchers.

In other words, this is the interface between data fetching and data analysis.

"""

import copy

import numpy as np
import pandas as pd


class ExperimentData(object):
	# TODO: allow definition of the name of 'entity': would be nicer if the index
	# itself maintained the name 'chash' or 'order_number' etc.
	# TODO: explain the mandatory_metadata in the exception raised
	mandatory_metadata = {'experiment', 'source'}
	primary_indices = ['entity', 'variant']
	optional_kpi_indices = ['time_since_treatment']
	known_feature_metrics = {
		'age',
		'zalando_age',
		'customer_age',
		'customer_zalando_age',
		'gender', 'sex',
		'feature',
		'treatment_start_time',
		'orders_existing',
		'orders_prev_year',
		'start_segment',
		# 'business_customer', #these SHOULD be features, but because we have no historical data, their value will depend on when we retrieve the data - i.e. they can change after the experiment.
		# 'corporate_customer',
		# 'special_customer',
		'existing_customer',
		'exposed_customer',
		'clv',
		# I know this is somewhat controversial, but I want to insist on using PCII as KPI and CLV as a feature always meaning 'sum of PCII over lifetime up to treatment start'
	}

	def __init__(self, metrics=None, metadata={}, features='default',
				 deepcopy=False):
		"""
    Want to be able to create results from just a single dataframe.

    Args:
        metrics: data frame that contains either KPI or feature
        metadata: the metadata dict
        features: either 'default', which searches the metrics data frame for predefined feature names
                  or list, which subsets the metrics data frame with the given column indices
                  or data frame, which is feature data frame itself and metrics is either KPI or None
                  or None
        deepcopy: the internal data frames are, by default, shallow copies of the input
                            dataframes: this means the actual data arrays underlying the frames are
                            references to the input. In most use-cases, this is desired (reindexing
                            will not reindex the original etc.) but it may have some edge-case issues.
    """

		self.metadata = metadata or {}

		feature_indices = copy.deepcopy(self.primary_indices)
		kpi_indices = copy.deepcopy(self.primary_indices)
		if metrics is not None:
			kpi_indices += [i for i in self.optional_kpi_indices if i in metrics.columns]

		if metrics is None:
			if not isinstance(features, pd.DataFrame):
				raise ValueError('No metrics provided!')
			else:
				self.kpis = pd.DataFrame(columns=self.primary_indices)
				self.features = features.copy(deep=deepcopy)
				self.variant_names = set(np.unique(self.features.variant))
		else:
			if isinstance(features, pd.DataFrame):
				self.kpis = metrics.copy(deep=deepcopy)
				self.features = features.copy(deep=deepcopy)
				self.variant_names = set(np.unique(self.features.variant))
			elif isinstance(features, list):
				if len(features) == 0:
					self.kpis = metrics.copy(deep=deepcopy)
					self.features = pd.DataFrame(columns=self.primary_indices)
					self.variant_names = set(np.unique(self.kpis.variant))
				else:
					self.kpis = metrics.drop(metrics.columns[features], axis=1)
					primary_idx = [metrics.columns.get_loc(x) for x in self.primary_indices]
					feature_idx = primary_idx + features
					self.features = metrics.iloc[:, feature_idx]
					self.variant_names = set(np.unique(self.features.variant))
			elif features == 'default':
				# TODO: use the detect_features function
				features_present = {m for m in metrics if
									m.lower() in self.known_feature_metrics | set(feature_indices)}
				kpis_present = {m for m in metrics if m.lower() not in self.known_feature_metrics}
				self.features = metrics.loc[:, features_present]
				self.kpis = metrics.loc[:, kpis_present]
				self.variant_names = set(np.unique(self.features.variant))
			elif features is None:
				self.kpis = metrics.copy(deep=deepcopy)
				self.features = pd.DataFrame(columns=self.primary_indices)
				self.variant_names = set(np.unique(self.kpis.variant))
			else:
				raise ValueError('Unknown features argument!')

		# validations
		missing_metadata = set(self.mandatory_metadata) - set(self.metadata)
		if len(missing_metadata) > 0:
			raise KeyError('ExperimentData requires metadata: [' + ','.join(missing_metadata) + ']')

		if 'primary_KPI' in self.metadata and (self.metadata['primary_KPI'] not in self.kpis):
			raise KeyError('ExperimentData requires the primary_KPI (\'{}\') to be present in the KPIs.'.format(
				self.metadata['primary_KPI']))

		if len(self.variant_names) < 2:
			raise KeyError('Less than 2 variants found!')

		self.features.set_index(list(feature_indices), inplace=True)
		self.kpis.set_index(list(kpi_indices), inplace=True)

		# check if time domain exists
		if 'time_since_treatment' in self.kpis.index.names:
			self.kpis_time = self.kpis
			# NOTE: for most current implemented KPIs, the sum aggregation is
			# appropriate
			self.kpis = self.kpis_time.groupby(level=['entity', 'variant']).sum()
		else:
			self.kpis_time = None

	@property
	def feature_names(self):
		"""List of features"""
		return set([] if self.features is None else self.features.columns)

	@property
	def kpi_names(self):
		"""List of KPIs"""
		return set([] if self.kpis is None else self.kpis.columns)

	# @property
	# def variant_names(self):
	# 	"""List of Variants"""
	# 	return set([] if self.features is None else set(self.features.index.get_level_values('variant')))

	@property
	def metric_names(self):
		"""List of metrics (KPIs and features)"""
		return self.feature_names.union(self.kpi_names)

	def __str__(self):
		return '{} \'{}\' with {:d} features and {:d} KPIs (primary: \'{}\'), {:d} entities'.format(
			self.__class__.__name__,
			self.metadata['experiment'],
			len(self.feature_names),
			len(self.kpi_names),
			self.metadata.get('primary_KPI', 'undefined'),
			self.features.index.nunique()  # TODO: should we explicitly count unique entities?
		)

	def __repr__(self):
		# TODO: improve this
		return 'ExperimentData(\nkpis={}\nfeatures={}\nmetadata={}'.format(
			repr(self.kpis),
			repr(self.features),
			repr(self.metadata),
		)

	@property
	def metrics(self):
		"""
    Simply joins the KPIs and the features.

    TODO: it may well be worth investigating caching this result because the
    features and kpis will rarely change, and accessing them in this way is likely to be common.
    """
		if 'time_since_treatment' in self.kpis.index.names:
			return self.kpis.reset_index('time_since_treatment').join(self.features).set_index('time_since_treatment',
																							   append=True)
		else:
			return self.kpis.join(self.features)

	def __getitem__(self, key):
		"""
    Allows indexing the ExperimentData directly as though it were a DataFrame
    composed of KPIs and Features.
    """
		return self.metrics.__getitem__(key)

	# Q: is it possible to pass a whole lot of functions to
	# a member variable without specifying each?
	def feature_boxplot(self, feature, kpi, **kwargs):
		self.metrics.set_index(feature, append=True).unstack(level=['variant', feature])[kpi].boxplot(**kwargs)


def detect_features(metrics):
	"""
  Automatically detect which of the metrics are features.
  """
	from warnings import warn

	if 'time_since_treatment' in metrics:
		# TODO: test this!
		# assuming that time is the only optional extra index for kpis...
		nunique = metrics.groupby(ExperimentData.primary_indices).agg(
			lambda x: len(x.unique())).max()
		features_present = (nunique == 1)
	# TODO: drop time dimension from features (i.e. take first value)
	else:
		features_present = {m for m in metrics if m.lower() in
							ExperimentData.known_feature_metrics}

	warn('not tested')

	return features_present


if __name__ == '__main__':
	from tests.tests_core.test_data import generate_random_data

	np.random.seed(0)
	metrics, meta = generate_random_data()
	D = ExperimentData(metrics, meta, 'default')
