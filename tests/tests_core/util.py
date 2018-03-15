import os
import pandas as pd


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
