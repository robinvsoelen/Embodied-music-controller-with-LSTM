"""
Script for merging data samples stored in  "DATA_TEMP_DIR".
"""

import pandas as pd
import glob

from config import *

def merge_pckls(directory):
    """
    Take as input a directory and merge all .pckl
    files in this directory.
    """
    dfs = []
    for file in glob.glob(directory + '*.pckl'):
        data = pd.read_pickle(file)
        data = data.dropna()
        dfs.append(data)
    

    try:
        df = pd.concat(dfs, ignore_index=True)
        if (directory == TRIGGER_DATA_TEMP_DIR):
            df.to_pickle(TRIGGER_DATA_PATH)
        else:
            df.to_pickle(DATA_PATH)
        return df
    except ValueError:
        raise ValueError

if __name__ == '__main__': # pragma: no cover
    try:
        df = merge_pckls(DATA_TEMP_DIR)
        print("Final data shape: ", df.shape)
    except ValueError:
        raise ValueError
