from os import rmdir, makedirs, getcwd, remove, walk
import unittest
from os.path import dirname, join, realpath, exists

import expan.data.csv_fetcher as csv_fetcher
import simplejson as json
import tests.tests_core.test_data as td

__location__ = realpath(join(getcwd(), dirname(__file__)))

TEST_FOLDER = join(__location__, 'test_folder')


class CsvFetcherTestCase(unittest.TestCase):
	def setUp(self):

		# create test folder
		if not exists(TEST_FOLDER):
			makedirs(TEST_FOLDER)

		# generate metrics and metadata
		(metrics, metadata) = td.generate_random_data()

		# save metrics to .csv.gz file in test folder
		metrics.to_csv(path_or_buf=join(TEST_FOLDER, 'metrics.csv.gz'), compression='gzip')

		# save metadata to .json file in test folder
		with open(join(TEST_FOLDER, 'metadata.json'), 'w') as f:
			json.dump(metadata, f)

	def tearDown(self):

		# remove all test files and test folder
		for root, dirs, files in walk(TEST_FOLDER, topdown=False):
			for name in files:
				remove(join(root, name))
			for name in dirs:
				rmdir(join(root, name))
		rmdir(TEST_FOLDER)

	def test_csv_fetcher(self):
		# should work:
		csv_fetcher.get_data(TEST_FOLDER)

		# should not work:
		with self.assertRaises(AssertionError):
			csv_fetcher.get_data(join(__location__, '..'))
