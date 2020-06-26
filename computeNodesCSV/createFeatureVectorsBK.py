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

def createColumnsWithDummyValues(df, old_df, numfeatures):
    for column in old_df.columns:
        if column != 'label':
            df['mean_' + column] = 1
            df['std_' +  column] = 1
            if(args.features == 11):
                df['perc5_' + column] = 1
            df['perc25_' + column] = 1
            df['perc75_' + column] = 1
            if(args.features == 11):
                df['perc95_' + column] = 1
            df['sumdiff_' +  column] = 1
            df['last_' + column] = 1
            if(args.features == 11):
                df['min_' + column] = 1
                df['max_' + column] = 1
                df['median_' + column] = 1
        else:
            df['label'] = 1
    return df


def setColumnStats(new_df, old_df, column, row, nwindow, numfeatures):
    if column != 'label':
        new_df.loc[row]['mean_' + column] = old_df[:nwindow][column].mean()
        new_df.loc[row]['std_' +  column] = old_df[:nwindow][column].std()
        if(args.features == 11):
            new_df.loc[row]['perc5_' + column] = old_df[:nwindow ][column].quantile(0.05)
        new_df.loc[row]['perc25_' + column] = old_df[:nwindow][column].quantile(0.25)
        new_df.loc[row]['perc75_' + column] = old_df[:nwindow][column].quantile(0.75)
        if(args.features == 11):
            new_df.loc[row]['perc95_' + column] = old_df[:nwindow][column].quantile(0.95)
        values = old_df[:nwindow][column].to_numpy()
        sumdiff_value = getSumChanges(values)
        new_df.loc[row]['sumdiff_' +  column] = sumdiff_value
        new_df.loc[row]['last_' + column] = old_df.loc[nwindow-1][column] #get last value
        if(args.features == 11):
            new_df.loc[row]['min_' + column] = old_df[:nwindow][column].min()
            new_df.loc[row]['max_' + column] = old_df[:nwindow][column].max()
            new_df.loc[row]['median_' + column] = old_df[:nwindow][column].median()
    else:
        values = old_df[:nwindow][column].to_numpy()
        label = getMostRecentFault(values)
        new_df.loc[row]['label'] = label

    #return new_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--timewindow", type=str, default='60s') 
    parser.add_argument("-f", "--features", type=int, default=6)
    parser.add_argument("-s", "--stepsize", type=int, default=1)
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
            new_df = createColumnsWithDummyValues(new_df, orig_df, args.features)
            #trick to have a moving rolling 60 seconds window of step size 1
            new_df_len_index = len(new_df) - (num_timewindow - 1)
            new_df = new_df[:new_df_len_index]
            #print(new_df)

            for row in tqdm(range(new_df_len_index)):
                #for each column, calculate the indicators defined in the LRZ report of March 2020
                for column in orig_df.columns:
                    setColumnStats(new_df, orig_df, column, row, num_timewindow, args.features)
                orig_df = orig_df.shift(-args.stepsize).dropna()

            #put label as last column
            label_col = new_df.pop('label')
            new_df['label'] = label_col

            print("Saving feature vectors for node " + file_entry.stem)
            new_df.to_csv(here.joinpath(file_entry.stem + f"_{args.features}f_{num_timewindow}s.csv"), index=False)
            print("Saving done")
