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
import re
import itertools
import time

#######functions to count lines of a file, from https://stackoverflow.com/a/27518377/1262118
def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )
########

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


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


"""
Get the sum of the changes within the array-like object; used to get
the sum of the changes within the aggregation window during rolling window operations.
"""
def getSumChanges(arr):
    return np.sum(np.diff(arr))

#orig_df = df[column]
#for each column, calculate the indicators defined in the LRZ report of March 2020 and the correlations between the columns
def calculateIndicators(column, orig_df, corrcolumns, window, step, numfeatures, compute_correlations):
    new_df = pd.DataFrame(index=orig_df.index)
    new_df['placeholder'] = 1
    new_df = new_df.rolling(window).mean().dropna()[::step].reset_index(drop=True)
    new_df.drop('placeholder', axis=1, inplace=True)
    if column != None:
        if column != 'label':
            new_df['mean_' + column] = orig_df[column].rolling(window).mean().dropna()[::step].reset_index(drop=True)
            new_df['std_' +  column] = orig_df[column].rolling(window).apply(lambda x: np.std(x, ddof=1)).dropna()[::step].reset_index(drop=True)
            if(numfeatures == 11):
                new_df['perc5_' + column] = orig_df[column].rolling(window).quantile(0.05).dropna()[::step].reset_index(drop=True)
            new_df['perc25_' + column] = orig_df[column].rolling(window).quantile(0.25).dropna()[::step].reset_index(drop=True)
            new_df['perc75_' + column] = orig_df[column].rolling(window).quantile(0.75).dropna()[::step].reset_index(drop=True)
            if(numfeatures == 11):
                new_df['perc95_' + column] = orig_df[column].rolling(window).quantile(0.95).dropna()[::step].reset_index(drop=True)
            new_df['sumdiff_' +  column] = orig_df[column].rolling(window).agg(getSumChanges).dropna()[::step].reset_index(drop=True)
            new_df['last_' + column] = orig_df[column].rolling(window).agg(lambda x: np.asarray(x)[-1]).dropna()[::step].reset_index(drop=True)
            if(numfeatures == 11):
                new_df['min_' + column] = orig_df[column].rolling(window).min().dropna()[::step].reset_index(drop=True)
                new_df['max_' + column] = orig_df[column].rolling(window).max().dropna()[::step].reset_index(drop=True)
                new_df['median_' + column] = orig_df[column].rolling(window).median().dropna()[::step].reset_index(drop=True)
        else:
            new_df['label'] = orig_df[column].rolling(window).agg(getMostRecentFault).dropna()[::step].reset_index(drop=True)

        if compute_correlations == True:
            for othercol in orig_df.columns:
                if othercol != column and str('corr_' + column + '_' + othercol) in corrcols:
                    correlation_column_other = orig_df[column].rolling(window).corr(orig_df[othercol]).iloc[window-1:][::step].reset_index(drop=True)
                    correlation_column_other.fillna(0, inplace=True)
                    correlation_column_other = correlation_column_other.replace([np.inf], +1)
                    correlation_column_other = correlation_column_other.replace([-np.inf], -1)
                    new_df['corr_' + column + '_' + othercol] = correlation_column_other

    return new_df

"""
Calculate the next index from which to start in the next iteration of the algorithm
"""
def calculateNextIndex(current_index, numchunk, numwindow, stepsize):
    tmp_line =  numchunk

    count = 0
    while(tmp_line >= numwindow):
        tmp_line = tmp_line - stepsize
        count = count + 1

    return (current_index + count*stepsize, count*stepsize)

"""
Set the list of correlation columns names, leaving label out of it
"""
def setNamesOfCorrelationColumns(lst, columns):
    if not lst:
        columns.remove('label')
        couples = list(itertools.product(columns, columns))
        for coup in couples:
            #get only corr(A,B), not corr(B,A); moreover do not take corr(A,A)
            if coup[0] != coup[1] and ("corr_" + coup[1] + "_" + coup[0]) not in lst:
                lst.append("corr_" + coup[0] + "_" + coup[1])


def setCpuMetricsAttribute(lst, columns):
    if not lst:
        regex = re.compile('cpu[0-9]+\/')
        #get columns in the form "cpuXX/<metricname>"
        cpucols = [col for col in columns if regex.match(col)]
        #get only the <metricname> in "cpuXX/<metricname>", there will be duplicates
        dupcpuatt = [re.sub(r"cpu[0-9]+\/", "", col) for col in cpucols]
        #remove duplicates by using a set
        cpuatt_set = set(dupcpuatt)
        #convert again to list and assign to lst
        lst.extend(list(cpuatt_set))

"""
Horizontally compute min, max, 25th percentile, 75th percentile and mean for cpu specific metrics

@input df: dataframe
@input m: metric
"""
def computeCpuSpecificMetrics(df, m):
    ndf = pd.DataFrame()
    regex = re.compile("cpu[0-9]+\/" + m)
    selected_columns = [col for col in df.columns if regex.match(col)]
    ndf["min_cpus/" + m] = df[selected_columns].min(axis=1)
    ndf["max_cpus/" + m] = df[selected_columns].max(axis=1)
    ndf["perc25_cpus/" + m] = df[selected_columns].quantile(q=0.25, axis=1)
    ndf["perc75_cpus/" + m] = df[selected_columns].quantile(q=0.75, axis=1)
    ndf["mean_cpus/" + m] = df[selected_columns].mean(axis=1)
    return ndf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--timewindow", type=int, default=60) 
    parser.add_argument("-f", "--features", type=int, default=6)
    parser.add_argument("-s", "--stepsize", type=int, default=1)
    parser.add_argument("-p", "--processes", type=int, default=4)
    parser.add_argument("-c", "--chunk", type=int, default=10000)
    parser.add_argument("-r", "--corr", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-l", "--horiz", type=str2bool, nargs='?', const=True, default=False)
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
            #get line count by counting how many newlines there are
            orig_numlines = rawgencount(filepath)
            #get row count, excluding the header
            orig_numlines = orig_numlines - 1
            print("File lines: ", orig_numlines)

            isFirst = True

            if os.path.exists(here.joinpath(file_entry.stem + f"_{args.features}f_{num_timewindow}s_{args.stepsize}step.csv")):
                os.remove(here.joinpath(file_entry.stem + f"_{args.features}f_{num_timewindow}s_{args.stepsize}step.csv"))

            pbar = tqdm(total=orig_numlines - num_timewindow + 1)
            currindex = 1
            corrcols = []
            cpu_metrics_att = []
            while(currindex < orig_numlines):
                orig_df = pd.read_csv(filepath, header=0, index_col=0, parse_dates=True, skiprows=range(1,currindex), nrows=args.chunk)
                #get attributes <metricname> contained in columns of the form "cpuXX/<metricname>"
                #print("setCpuMetricsAttribute called")
                if(args.horiz==True):
                    setCpuMetricsAttribute(cpu_metrics_att, orig_df.columns)
                #print("setCpuMetricsAttribute finished")

                #starting_time = time.time()
                #print("Horizontally computing cpu specific metrics")
                #orig_df = computeCpuSpecificMetrics(orig_df, cpu_metrics_att)
                if(args.horiz==True):
                    for groupm in list(grouper(args.processes, cpu_metrics_att)):
                        arg_list = []
                        for m in groupm:
                            if m != None:
                                arg_list.append((orig_df, m))
                        with multiprocessing.Pool(processes=args.processes) as pool:
                            results = pool.starmap(computeCpuSpecificMetrics, arg_list)
                        results.insert(0, orig_df)
                        orig_df = pd.concat(results, axis=1)
                        results = []
                #ending_time = time.time()
                #print("Execution time in seconds: ", ending_time - starting_time)


                #remove columns with name "cpuXX/<metricname>"
                orig_df = orig_df.loc[:, ~orig_df.columns.str.contains('cpu[0-9]')]
                #drop applicationLabel and faultPred
                orig_df.drop(['experiment/applicationLabel', 'faultPred'], axis=1, inplace=True)
                #rename faultLabel in label
                orig_df.rename(columns={'faultLabel' : 'label'}, inplace=True)
                orig_df.reset_index(drop=True, inplace=True)

                if args.corr == True:
                    #print("setNamesOfCorrelationColumns called")
                    #set corrcols list only once, when the list is empty
                    setNamesOfCorrelationColumns(corrcols, list(orig_df.columns))
                    #print("setNamesOfCorrelationColumns finished")

                #initialize new dataframe to empty timeseries
                new_df = pd.DataFrame(index=orig_df.index)
                new_df['placeholder'] = 1
                #group by with a window of 60s, the column placeholder is just a trick to make groupby work, we can remove it afterwards
                new_df = new_df.rolling(num_timewindow).mean().dropna()[::args.stepsize].reset_index(drop=True)
                new_df.drop('placeholder', axis=1, inplace=True)

                #For each column, calculate the indicators defined in the LRZ report of March 2020.
                #This is done using multiprocessing, each process addresses a column.
                #First we group columns in group of n (where n is the number of processes)
                for groupcol in list(grouper(args.processes, orig_df.columns)):
                    #create a list of tuples, where each tuple is made by (column_name, df[column], ...otherarguments)
                    arg_list = []
                    for elem in groupcol:
                        if elem != None:
                            arg_list.append((elem, orig_df, corrcols, num_timewindow, args.stepsize, args.features, args.corr))
                    with multiprocessing.Pool(processes=args.processes) as pool:
                        results = pool.starmap(calculateIndicators, arg_list)
                    results.insert(0, new_df)
                    new_df = pd.concat(results, axis=1)
                    results = []

                    
                #reorder columns lexycographically
                #new_df = new_df.reindex(sorted(new_df.columns), axis=1)

                #put label as last column
                label_col = new_df.pop('label')
                new_df['label'] = label_col

                new_df.to_csv(here.joinpath(file_entry.stem + f"_{args.features}f_{num_timewindow}s_{args.stepsize}step.csv"), mode='a', index=False, header=isFirst)
                isFirst = False

                currindex, counter = calculateNextIndex(currindex, args.chunk, num_timewindow, args.stepsize)
                pbar.update(counter)
            pbar.close()
