import unittest
import numpy as np
import expan.core.util as util


class UtilFunctionsTestCase(unittest.TestCase):
    def test_find_value_by_key_with_condition(self):
        list_of_dicts = [{'bla': 1, 'blu': 2},
                         {'bla': 3, 'blu': 4},
                         {'bla': 5, 'blu': 6}]
        self.assertEqual(util.find_value_by_key_with_condition(list_of_dicts, 'bla', 5, 'blu'), 6)

        with self.assertRaises(IndexError):
            util.find_value_by_key_with_condition(list_of_dicts, 'bla', 'value_not_exist', 'blu')
        with self.assertRaises(KeyError):
            util.find_value_by_key_with_condition(list_of_dicts, 'bla', 5, 'key_not_exist')

    def test_is_nan(self):
        a = float('nan')   # NaN
        self.assertTrue(util.is_nan(a))
        b = None
        self.assertFalse(util.is_nan(b))
        c = 2
        self.assertFalse(util.is_nan(c))
        d = 'str'
        self.assertFalse(util.is_nan(d))

    def test_drop_nan(self):
        nan = float('nan')   # NaN

        array_with_nan_1d = np.array([nan, nan, 1, 2, 3])
        returned_array_1d = util.drop_nan(array_with_nan_1d)
        self.assertTrue(all(returned_array_1d == np.array([1, 2, 3])))

        array_with_nan_2d = np.array([[nan], [nan], [1], [2], [3]])
        returned_array_2d = util.drop_nan(array_with_nan_2d)
        print(returned_array_2d)
        self.assertTrue(all(returned_array_2d == np.array([[1], [2], [3]])))

        array_without_nan_1d = np.array([1, 2])
        returned_array = util.drop_nan(array_without_nan_1d)
        self.assertTrue(all(returned_array == np.array([1, 2])))


if __name__ == '__main__':
    unittest.main()
