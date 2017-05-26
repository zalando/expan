import os

import pandas as pd


def get_norm_temp_data(fdir, fname='normtemp.dat.txt.gz'):
    """
    normtemp.dat.txt.gz:
          Data retrieved on 2015/02/18 from:
        http://www.amstat.org/publications/jse/jse_data_archive.htm
      """

    # Read data from csv to pd.dataFrame
    data = pd.read_csv(
        os.path.join(fdir, fname),
        delim_whitespace=True,
        header=None,
        skip_blank_lines=True,
        names=['temperature', 'gender', 'heartrate'],
    )

    # Return the pd.dataFrame
    return data


def get_framingham_data(fdir, fname='framingham_heart_study_exam7.csv'):
    """
    framingham_heart_study_exam7.csv:
        Data retrieved on 2015/10/28 from:
        http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals5.html
      """

    # Read data from csv to pd.dataFrame
    data = pd.read_csv(os.path.join(fdir, fname),
                       header=[0, 1],
                       index_col=0)
    # Set index
    data.index.name = 'metric'

    # Read data from csv to pd.dataFrame
    return data
