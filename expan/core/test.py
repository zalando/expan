'''import sys, os
from os.path import dirname, join, realpath
sys.path.insert(0, join(os.getcwd(), 'tests'))
import json
import numpy as np
from expan.core.util import generate_random_data

np.random.seed(0)
data,metadata = generate_random_data()

import expan
exp = expan.experiment.Experiment('A', data, metadata, report_kpi_names='normal_shifted', derived_kpis='normal_unequal_variance')


report_kpi_names = 'normal_shifted'
report_kpi_names = ['normal_shifted']

derived_kpis='normal_unequal_variance'

derived_kpi_names    = [k['name']    for k in derived_kpis]
derived_kpi_formulas = [k['formula'] for k in derived_kpis]


derived_kpis=[{'name':'vale', 'formula':'vale'}]

for i in derived_kpis:
    if not isinstance(i, dict):
        raise TypeError('Derived kpis should be an array of dictionaries')
    if 'name' not in i or 'formula' not in i:
        raise ValueError('Dictionaries should have keys "name" and "formula"')'''
