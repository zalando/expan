# import os
import unittest

import numpy as np

import expan.core.util as util


class UtilFunctionsTestCase(unittest.TestCase):
	def test_scale_range(self):
		res = util.scale_range([1, 3, 5])
		np.testing.assert_allclose(res, [0., 0.5, 1.])

		res = util.scale_range([1, 2, 3, 4, 5])
		np.testing.assert_allclose(res, [0., 0.25, 0.5, 0.75, 1.])

		res = util.scale_range([1, 3, 5, np.inf])
		np.testing.assert_allclose(res, [0., 0.5, 1., np.inf])

		res = util.scale_range([1, 3, 5, -np.inf])
		np.testing.assert_allclose(res, [0., 0.5, 1., -np.inf])

		res = util.scale_range([1, 3, 5, -np.inf], squash_inf=True)
		np.testing.assert_allclose(res, [0., 0.5, 1., 0.])

		res = util.scale_range([1, 3, 5, np.inf], squash_inf=True)
		np.testing.assert_allclose(res, [0., 0.5, 1., 1.])

		res = util.scale_range([1, 3, 5], new_min=0.5)
		np.testing.assert_allclose(res, [0.5, 0.75, 1.])

		res = util.scale_range([1, 3, 5], old_min=1, old_max=4)
		np.testing.assert_allclose(res, [0., 0.66666667, 1.])

		res = util.scale_range([5], old_max=4)
		np.testing.assert_allclose(res, [1.])


if __name__ == '__main__':
	unittest.main()
