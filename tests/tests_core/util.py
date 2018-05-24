import os
import pandas as pd
import numpy as np


def get_norm_temp_data(fdir, fname='normtemp.dat.txt.gz'):
    """ Gets normtemp dataset: normtemp.dat.txt.gz.
    Data retrieved on 2015/02/18 from: http://www.amstat.org/publications/jse/jse_data_archive.htm

    :param fdir: directory containing the data
    :type  fdir: str
    :param fname: data file name
    :type  fname: str
    
    :return data frame containing test normtemp data
    :rtype  pd.DataFrame
    """

    # Read data from csv to pd.dataFrame
    data = pd.read_csv(
        os.path.join(fdir, fname),
        delim_whitespace=True,
        header=None,
        skip_blank_lines=True,
        names=['temperature', 'gender', 'heartrate'],
    )
    return data


def get_framingham_data(fdir, fname='framingham_heart_study_exam7.csv'):
    """ Gets framingham heart study dataset: framingham_heart_study_exam7.csv:
    Data retrieved on 2015/10/28 from:
    http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals5.html

    :param fdir: directory containing the data
    :type  fdir: str
    :param fname: data file name
    :type  fname: str
    
    :return data frame containing test framingham heart study dataset
    :rtype  pd.DataFrame
    """
    data = pd.read_csv(os.path.join(fdir, fname),
                       header=[0, 1],
                       index_col=0)
    data.index.name = 'metric'
    return data


def get_test_data_revenue_order():
    """ Small test data for testing with kpis revenue and number of orders.
    
    :return data frame containing test dataset
    :rtype  pd.DataFrame
    """
    test_data = np.array([['fullVisitorId', 'visitorId', 'date', 'entity', 'variant',
                           'revenue', 'orders', 'appdomain', 'device_type', 'gender'],
                           ['fv1', 'v1', '2017-10-01', 'client-1', 'variant1', 2.3, 4.5, 'AT', 'mobile', 'female'],
                           ['fv1', 'v2', '2017-10-01', 'client-1', 'variant1', 1.2, 0.5, 'AT', 'desktop', 'female'],
                           ['fv1', 'v1', '2017-10-01', 'client-2', 'variant1', 2.1, 3.5, 'AT', 'mobile', 'male'],
                           ['fv1', 'v2', '2017-10-01', 'client-2', 'variant1', 1.1, 0.7, 'AT', 'desktop', 'male'],
                           ['fv1', 'v1', '2017-10-01', 'client-3', 'variant2', 2.3, 4.5, 'AT', 'mobile', 'male'],
                           ['fv1', 'v2', '2017-10-01', 'client-3', 'variant2', 1.2, 0.5, 'AT', 'desktop', 'male'],
                           ['fv1', 'v1', '2017-10-01', 'client-4', 'variant2', 2.1, 3.5, 'AT', 'mobile', 'female'],
                           ['fv1', 'v2', '2017-10-01', 'client-4', 'variant2', 1.1, 0.7, 'AT', 'desktop',
                            'female']])
    return pd.DataFrame(data=test_data[1:, 1:], columns=test_data[0, 1:]).convert_objects(convert_numeric=True)
