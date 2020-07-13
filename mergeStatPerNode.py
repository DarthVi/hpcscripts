"""
2020-06-22 17:18
@author: Vito Vincenzo Covella
"""

import pandas as pd
import numpy as np
import os
import argparse
import pathlib
from pathlib import Path
import re
from tqdm import tqdm
import time

"""
Check if dataframe is monotone. If it is, convert it to the delta equivalent and interpolate
missing values
"""
def monotonicityCheck(df, interp_method):
    firstColumn = df.columns[0]
    if (df[firstColumn].is_monotonic_increasing or df[firstColumn].is_monotonic_decreasing) and ('applicationLabel' not in firstColumn and 'faultLabel' not in firstColumn and 'faultPred' not in firstColumn):
        df = df.diff()
        df = df.interpolate(method=interp_method, axis=0, limit_direction='backward')
    return df

def transformCSV(csv):
    #transform the Time column to a timeseries in nanoseconds
    csv['Time'] = pd.to_datetime(csv['Time'], unit='ns')
    #rename the Value column with the Sensor name (it's the same for all rows)
    #the regex removes the Sensor's "path", which is the same for all rows
    csv = csv.rename(columns={'Value': re.sub('(\/[^\/]+\/){1}([^\/]+\/){3}', '', csv.iloc[0]['Sensor'])})
    csv = csv.drop('Sensor', axis=1)
    csv = csv.set_index('Time')
    #truncate the indices to get the seconds
    csv.index = csv.index.floor('s')
    #remove duplicates by taking the mean
    csv = csv.groupby(csv.index).mean()
    return csv

"""
Fills the NaN values of specific columns in forward and backward way
"""
def fillLabelNA(df, column):
    if column in df.columns:
        df[column] = df[column].ffill().bfill()
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interpolationmethod", type=str, default='linear')
    parser.add_argument("-t", "--threshold", type=int, default=200) 
    args = parser.parse_args()
    #path of the folder in which this script is located
    here = pathlib.Path(__file__).parent

    
    with open(here.joinpath("compute_nodes.txt")) as compnodes_file:
        nodes = list(map(str.strip, compnodes_file))

    main_df = None
    labelCol = ['experiment/applicationLabel', 'faultPred', 'faultLabel']

    for node in nodes:
        print("Processing node ", node)
        csvlist = []
        dflist = []

        for file_entry in here.iterdir():
            if file_entry.is_file() and file_entry.suffix == '.csv' and node in file_entry.name:
                csvlist.append(here.joinpath(file_entry.name))

        main_df = pd.read_csv(csvlist[0], header=0)
        main_df = transformCSV(main_df)
        main_df = monotonicityCheck(main_df, args.interpolationmethod)

        for csv in tqdm(csvlist[1:]):
            second_df = pd.read_csv(csv, header=0)
            second_df = transformCSV(second_df)
            second_df = monotonicityCheck(second_df, args.interpolationmethod)
            #align the two CSV on the Time index
            main_df, second_df = main_df.align(second_df, axis=0)

            dflist.append(second_df)
            
            if(len(dflist) >= args.threshold):
                dflist.insert(0, main_df)
                main_df = pd.concat(dflist, axis=1)
                dflist = []
            #main_df = pd.merge(main_df, second_df, left_index=True, right_index=True)

        #if append list is not empty (check if there are trailing dataframe within the list which is however shorter than threshold)
        if dflist:
            dflist.insert(0, main_df)
            main_df = pd.concat(dflist, axis=1)
            dflist = []


        print("Interpolating")
        start_time = time.time()
        #check if label columns are presente and fills NaN values backward and forward
        for col in labelCol:
            main_df = fillLabelNA(main_df, col)

        #replace NaN values with interpolated ones (along the column)
        main_df.interpolate(method=args.interpolationmethod, axis=0, inplace=True)
        #fill NaN in first row if present
        main_df.bfill(inplace=True)
        end_time = time.time()
        print("Execution of last step in seconds: ", end_time - start_time)

        print("Saving " + node + ".csv" + " file")
        start_time = time.time()
        main_df.to_csv(here.joinpath("computeNodesCSV/" + node + ".csv"))
        print("Saving done")
        end_time = time.time()
        print("Execution of last step in seconds: ", end_time - start_time)
