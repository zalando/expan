"""
CSV fetcher
Expects as input a folder containing the following files:
 - one .csv or .csv.gz with 'metrics' in the filename
 - one .txt containing 'metadata' in the filename

Opens the files and uses them to create an ExperimentData object which it then returns.
"""

from os import listdir
from os.path import isfile, join

import simplejson as json

from expan.core.experimentdata import *


def get_data(folder_path):
	files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

	try:
		assert ('metrics' in '-'.join(files))
		assert ('metadata' in '-'.join(files))

		metrics = metadata = None

		for f in files:

			if 'metrics' in f:
				try:
					metrics = pd.read_csv(folder_path + '/' + f)
				except Exception as e:
					print
					e

			elif 'metadata' in f:
				try:
					with open(folder_path + '/' + f, 'r') as input_json:
						metadata = json.load(input_json)
				except ValueError as e:
					print
					e
					raise

		return ExperimentData(metrics=metrics, metadata=metadata)

	except AssertionError as e:
		print
		e
		raise
