"""
2020-06-24 16:31

@author: Vito Vincenzo Covella
"""

import os
import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def getMostRecentFault(arr):
    #if everything is labeled with "healthy" (label 0), the sum is 0, return 0
    if np.sum(arr) == 0:
        return 0
    else:#else return the most recent fault label
        currentLabel = 0
        for label in arr:
            if label != 0:
                currentLabel = label
        return currentLabel

"""
Get the sum of the changes within the array-like object; used to get
the sum of the changes within the aggregation window during resampling
"""
def getSumChanges(arr):
    number_of_different_values = len(np.unique(arr))
    if number_of_different_values == 1:
        return 0
    else:
        #if there are n different values, then the value changes n - 1 times
        return number_of_different_values - 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--timewindow", type=str, default='60s') 
    parser.add_argument("-f" "--features", type=int, default=6)
    args = parser.parse_args()

    if args.features != 6 and args.feature != 11:
        exit(1)

    timewindow = args.timewindow
    num_timewindow = int(timewindow[:-1])

    here = pathlib.Path(__file__).parent

    for file_entry in here.iterdir():
        if file_entry.is_file() and file_entry.suffix == '.csv':
            filepath = here.joinpath(file_entry.name)
            orig_df = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True)
            #remove columns with name "cpuXX/<metricname>"
            orig_df = orig_df.loc[:, ~orig_df.columns.str.contains('cpu[0-9]')]
            #drop applicationLabel and faultPred
            orig_df.drop(['experiment/applicationLabel', 'faultPred'], axis=1, inplace=True)
            #rename faultLabel in label
            orig_df.rename(columns={'faultLabel' : 'label'}, inplace=True)
            orig_df.reset_index(drop=True, inplace=True)
            #initialize new dataframe to empty timeseries
            new_df = pd.DataFrame(index=orig_df.index)
            new_df['placeholder'] = 1
            #group by with a window of 60s, the column placeholder is just a trick to make groupby work, we can remove it afterwards
            new_df = new_df.groupby(res_df.index // num_timewindow).mean()
            new_df.drop('placeholder', axis=1, inplace=True)

            for column in tqdm(orig_df.columns):
                if column != 'label':
                    new_df['mean_' + column] = orig_df[column].groupby(res_df.index // 60).mean()
                    new_df['std_' +  column] = orig_df[column].groupby(res_df.index // 60).std()#TODO:Completare il codice
                else:
