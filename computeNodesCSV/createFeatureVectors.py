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
import multiprocessing
import itertools
from re import search

"""grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

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
the sum of the changes within the aggregation window during rolling window operations.
"""
def getSumChanges(arr):
    return np.sum(np.diff(arr))

#orig_df = df[column]
#for each column, calculate the indicators defined in the LRZ report of March 2020
def calculateIndicators(column, orig_df, window, step, numfeatures):
    new_df = pd.DataFrame(index=orig_df.index)
    new_df['placeholder'] = 1
    new_df = new_df.rolling(window).mean()[::step].dropna().reset_index(drop=True)
    new_df.drop('placeholder', axis=1, inplace=True)
    if column != None:
        if search("cpu[0-9]", column):
            if(numfeatures == 11):
                new_df['perc5_' + column] = orig_df.rolling(window).quantile(0.05)[::step].dropna().reset_index(drop=True)
            new_df['perc25_' + column] = orig_df.rolling(window).quantile(0.25)[::step].dropna().reset_index(drop=True)
            new_df['perc75_' + column] = orig_df.rolling(window).quantile(0.75)[::step].dropna().reset_index(drop=True)
            if(numfeatures == 11):
                new_df['perc95_' + column] = orig_df.rolling(window).quantile(0.95)[::step].dropna().reset_index(drop=True)
        elif column != 'label':
            new_df['mean_' + column] = orig_df.rolling(window).mean()[::step].dropna().reset_index(drop=True)
            new_df['std_' +  column] = orig_df.rolling(window).std()[::step].dropna().reset_index(drop=True)
            if(numfeatures == 11):
                new_df['perc5_' + column] = orig_df.rolling(window).quantile(0.05)[::step].dropna().reset_index(drop=True)
            new_df['perc25_' + column] = orig_df.rolling(window).quantile(0.25)[::step].dropna().reset_index(drop=True)
            new_df['perc75_' + column] = orig_df.rolling(window).quantile(0.75)[::step].dropna().reset_index(drop=True)
            if(numfeatures == 11):
                new_df['perc95_' + column] = orig_df.rolling(window).quantile(0.95)[::step].dropna().reset_index(drop=True)
            new_df['sumdiff_' +  column] = orig_df.rolling(window).agg(getSumChanges)[::step].dropna().reset_index(drop=True)
            new_df['last_' + column] = orig_df.rolling(window).agg(lambda x: np.asarray(x)[-1])[::step].dropna().reset_index(drop=True)
            if(numfeatures == 11):
                new_df['min_' + column] = orig_df.rolling(window).min()[::step].dropna().reset_index(drop=True)
                new_df['max_' + column] = orig_df.rolling(window).max()[::step].dropna().reset_index(drop=True)
                new_df['median_' + column] = orig_df.rolling(window).median()[::step].dropna().reset_index(drop=True)
        else:
            new_df['label'] = orig_df.rolling(window).agg(getMostRecentFault)[::step].dropna().reset_index(drop=True)

    return new_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--timewindow", type=int, default=60) 
    parser.add_argument("-f", "--features", type=int, default=6)
    parser.add_argument("-s", "--stepsize", type=int, default=1)
    parser.add_argument("-p", "--processes", type=int, default=4)
    args = parser.parse_args()

    if args.features != 6 and args.features != 11:
        print("Wrong arguments for features")
        exit(1)

    num_timewindow = args.timewindow

    here = pathlib.Path(__file__).parent

    for file_entry in here.iterdir():
        if file_entry.is_file() and file_entry.suffix == '.csv' and '_' not in file_entry.name:
            filepath = here.joinpath(file_entry.name)
            print("Processing file " + file_entry.name)
            orig_df = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True)
            #remove columns with name "cpuXX/<metricname>"
            #orig_df = orig_df.loc[:, ~orig_df.columns.str.contains('cpu[0-9]')]
            #drop applicationLabel and faultPred
            orig_df.drop(['experiment/applicationLabel', 'faultPred'], axis=1, inplace=True)
            #rename faultLabel in label
            orig_df.rename(columns={'faultLabel' : 'label'}, inplace=True)
            orig_df.reset_index(drop=True, inplace=True)
            #initialize new dataframe to empty timeseries
            new_df = pd.DataFrame(index=orig_df.index)
            new_df['placeholder'] = 1
            #group by with a window of 60s, the column placeholder is just a trick to make groupby work, we can remove it afterwards
            new_df = new_df.rolling(num_timewindow).mean()[::args.stepsize].dropna().reset_index(drop=True)
            new_df.drop('placeholder', axis=1, inplace=True)

            #For each column, calculate the indicators defined in the LRZ report of March 2020.
            #This is done using multiprocessing, each process addresses a column.
            #First we group columns in group of n (where n is the number of processes)
            for groupcol in tqdm(list(grouper(args.processes, orig_df.columns))):
                #create a list of tuples, where each tuple is made by (column_name, df[column], ...otherarguments)
                arg_list = []
                for elem in groupcol:
                    if elem != None:
                        arg_list.append((elem, orig_df[elem], num_timewindow, args.stepsize, args.features))
                with multiprocessing.Pool(processes=args.processes) as pool:
                    results = pool.starmap(calculateIndicators, arg_list)
                for dataframe in results:
                    if dataframe.empty == False:
                        new_df = pd.concat([new_df, dataframe], axis=1)
                
            #reorder columns lexycographically
            #new_df = new_df.reindex(sorted(new_df.columns), axis=1)

            #put label as last column
            label_col = new_df.pop('label')
            new_df['label'] = label_col
            #new_df.dropna(inplace=True)

            print("Saving feature vectors for node " + file_entry.stem)
            new_df.to_csv(here.joinpath(file_entry.stem + f"_{args.features}f_{num_timewindow}s_{args.stepsize}step.csv"), index=False)
            print("Saving done")
