from __future__ import division
from expan.core.experiment import Experiment
import pandas as pd
import numpy as np

def test_quantile_filtering():
    exp = Experiment({})
    df = pd.DataFrame.from_dict({   'earnings' : np.array([0,0,1,2]) / np.array([0,0,1,1]) })

    flags = exp._quantile_filtering(df, ['earnings'], 90, 'upper')
    assert flags.tolist() == [False, False, False, True]

    flags = exp._quantile_filtering(df, ['earnings'], 10, 'lower')
    assert flags.tolist() == [False, False, True, False]
