import logging
from os import listdir
from os.path import isfile, join

import pandas as pd
import simplejson as json

from expan.core.experiment import Experiment

logger = logging.getLogger(__name__)


def get_data(folder_path):
    """ Expects as input a folder containing the following files:

    - one .csv or .csv.gz with 'data' in the filename
    - one .json containing 'metadata' in the filename

    Opens the files and uses them to create an Experiment object which it then returns.

    :param folder_path: path to the Experiment data
    :type  folder_path: str
    :return: Experiment object with data
    :rtype:  Experiment
    """
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    try:
        assert ('data' in '-'.join(files))
        assert ('metadata' in '-'.join(files))
        data = metadata = None
        for f in files:
            if 'metadata' in f:
                with open(join(folder_path, f), 'r') as input_json:
                    metadata = json.load(input_json)
            elif 'data' in f:
                data = pd.read_csv(join(folder_path, f))
        return data, metadata

    except AssertionError as e:
        logger.error("An error occurred when fetching data from csv file.")
        raise e
