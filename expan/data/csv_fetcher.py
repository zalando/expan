"""CSV fetcher module.
"""

import logging
from os import listdir
from os.path import isfile, join

import pandas as pd
import simplejson as json

from expan.core.experiment import Experiment

logger = logging.getLogger(__name__)

def get_data(controlVariantName, folder_path):
    """
    Expects as input a folder containing the following files:
     - one .csv or .csv.gz with 'metrics' in the filename
     - one .txt containing 'metadata' in the filename

    Opens the files and uses them to create an Experiment object which it then returns.

    Args:
        folder_path:

    Returns:
        Experiment: Experiment object with loaded csv data

    """
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    try:
        assert ('metrics' in '-'.join(files))
        assert ('metadata' in '-'.join(files))

        metrics = metadata = None

        for f in files:

            if 'metrics' in f:
                metrics = pd.read_csv(join(folder_path, f))

            elif 'metadata' in f:
                with open(join(folder_path, f), 'r') as input_json:
                    metadata = json.load(input_json)

        return Experiment(controlVariantName, metrics, metadata)

    except AssertionError as e:
        logger.error("An error occured when fetching data from csv file.")
        raise e
