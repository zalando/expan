import numpy as np
import pandas as pd


def scale_range(x, new_min=0.0, new_max=1.0, old_min=None, old_max=None, squash_outside_range=True, squash_inf=False, ):
	"""
  Scales a sequence to fit within a new range.

  If squash_inf is set, then infinite values will take on the
  extremes of the new range (as opposed to staying infinite).

  NB: Infinity in the input is disregarded in the construction of
  the scale of the mapping.

  >>> scale_range([1,3,5])
  array([ 0. ,  0.5,  1. ])

  >>> scale_range([1,2,3,4,5])
  array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

  >>> scale_range([1,3,5, np.inf])
  array([ 0. ,  0.5,  1. ,  inf])

  >>> scale_range([1,3,5, -np.inf])
  array([ 0. ,  0.5,  1. , -inf])

  >>> scale_range([1,3,5, -np.inf], squash_inf=True)
  array([ 0. ,  0.5,  1. ,  0. ])

  >>> scale_range([1,3,5, np.inf], squash_inf=True)
  array([ 0. ,  0.5,  1. ,  1. ])

  >>> scale_range([1,3,5], new_min=0.5)
  array([ 0.5 ,  0.75,  1.  ])

  >>> scale_range([1,3,5], old_min=1, old_max=4)
  array([ 0.        ,  0.66666667,  1.        ])

  >>> scale_range([5], old_max=4)
  array([ 1.])

  """
	if squash_inf and not squash_outside_range:
		# TODO: "warn" is an unresolved reference
		warn(ValueError(
			'Makes no sense to squash infinity but not other numbers outside the source range. Will squash all outside range.'))
		squash_outside_range = True

	if isinstance(x, list):
		x = np.array(x)

	if old_min is None:
		old_min = np.min(x[~np.isinf(x)])
	if old_max is None:
		old_max = np.max(x[~np.isinf(x)])
	old_range = old_max - old_min
	new_max = float(new_max)
	new_min = float(new_min)
	new_range = new_max - new_min

	retval = (new_range * (x - old_min) / old_range) + new_min
	if squash_inf:
		retval[np.isinf(x) & (x > 0)] = new_max
		retval[np.isinf(x) & (x < 0)] = new_min

	if squash_outside_range:
		retval[~np.isinf(x) & (x > old_max)] = new_max
		retval[~np.isinf(x) & (x < old_min)] = new_min

	return retval


def reindex(df, axis=0):
	"""
  partial fulfilment of:
  https://github.com/pydata/pandas/issues/2770

  TODO: test.
  TODO: incorporate in pandas in drop() call and issue pull request
  """

	if axis not in [0, 1, 'index', 'columns']:
		raise NotImplementedError('only index and columns can be selected for axis')

	axis = 0 if axis in [0, 'index'] else 1
	axis_index = df.index if axis == 0 else df.columns  # TODO: fix for panels?

	new_index = pd.MultiIndex.from_tuples(axis_index.values)

	if axis == 0:
		df.index = new_index
	elif axis == 1:
		df.columns = new_index

	return df


if __name__ == '__main__':
	import doctest

	doctest.testmod()
