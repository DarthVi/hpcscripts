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
    return np.sum(np.diff(arr))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--timewindow", type=str, default='60s') 
    parser.add_argument("-f", "--features", type=int, default=6)
    args = parser.parse_args()

    if args.features != 6 and args.features != 11:
        print("Wrong arguments for features")
        exit(1)

    timewindow = args.timewindow
    num_timewindow = int(timewindow[:-1])

    here = pathlib.Path(__file__).parent

    for file_entry in here.iterdir():
        if file_entry.is_file() and file_entry.suffix == '.csv':
            filepath = here.joinpath(file_entry.name)
            print("Processing file " + file_entry.name)
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
            new_df = new_df.groupby(new_df.index // num_timewindow).mean()
            new_df.drop('placeholder', axis=1, inplace=True)

            #for each column, calculate the indicators defined in the LRZ report of March 2020
            for column in tqdm(orig_df.columns):
                if column != 'label':
                    new_df['mean_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).mean()
                    new_df['std_' +  column] = orig_df[column].groupby(orig_df.index // num_timewindow).std()
                    if(args.features == 11):
                        new_df['perc5_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).quantile(0.05)
                    new_df['perc25_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).quantile(0.25)
                    new_df['perc75_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).quantile(0.75)
                    if(args.features == 11):
                        new_df['perc95_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).quantile(0.95)
                    new_df['sumdiff_' +  column] = orig_df[column].groupby(orig_df.index // num_timewindow).agg(getSumChanges)
                    new_df['last_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).last()
                    if(args.features == 11):
                        new_df['min_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).min()
                        new_df['max_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).max()
                        new_df['median_' + column] = orig_df[column].groupby(orig_df.index // num_timewindow).median()
                else:
                    new_df['label'] = orig_df[column].groupby(orig_df.index // num_timewindow).agg(getMostRecentFault)

            #put label as last column
            label_col = new_df.pop('label')
            new_df['label'] = label_col

            print("Saving feature vectors for node " + file_entry.stem)
            new_df.to_csv(here.joinpath(file_entry.stem + f"_{args.features}f_{num_timewindow}s.csv"), index=False)
            print("Saving done")
