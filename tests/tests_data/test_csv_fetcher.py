import os
import unittest
from os.path import dirname, join, realpath

import simplejson as json

import expan.data.csv_fetcher as csv_fetcher
import tests.tests_core.test_data as td

__location__ = realpath(join(os.getcwd(), dirname(__file__)))

TEST_FOLDER = __location__ + 'test_folder'


class CsvFetcherTestCase(unittest.TestCase):
	def setUp(self):

		# create test folder
		if not os.path.exists(TEST_FOLDER):
			os.makedirs(TEST_FOLDER)

		# generate metrics and metadata
		(metrics, metadata) = td.generate_random_data()

		# save metrics to .csv.gz file in test folder
		metrics.to_csv(path_or_buf=TEST_FOLDER + '/metrics.csv.gz', compression='gzip')

		# save metadata to .json file in test folder
		with open(TEST_FOLDER + '/metadata.json', 'w') as f:
			json.dump(metadata, f)

	def tearDown(self):

		# remove all test files and test folder
		for root, dirs, files in os.walk(TEST_FOLDER, topdown=False):
			for name in files:
				os.remove(os.path.join(root, name))
			for name in dirs:
				os.rmdir(os.path.join(root, name))
		os.rmdir(TEST_FOLDER)

	def test_csv_fetcher(self):
		# should work:
		csv_fetcher.get_data(TEST_FOLDER)

		# should not work:
		with self.assertRaises(AssertionError):
			csv_fetcher.get_data(__location__ + '/../')
