"""
2020-10-16 12:52
@author: Vito Vincenzo Covella
"""

import numpy as np
import pandas as pd
import pathlib
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import OrderedDict
import argparse
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from FileFeatureReader.featurereaders import RFEFeatureReader
from FileFeatureReader.featurereader import FeatureReader
from utils import plot_heatmap, saveresults, majority_mean, str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="data/", help="Path in which there are the files to analyze")
    parser.add_argument("-r", "--savepath", type=str, default="results/", help="Relative path in which to store the results")
    parser.add_argument("-s", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-f", "--shuffle", type=str2bool, nargs='?', const=True, default=False, help="yes to shuffle the data, no otherwise")
    args = parser.parse_args()

    random.seed(42)

    if(args.sampling == 'majority'):
        sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    elif(args.sampling == 'majority_mean'):
        sampler = RandomUnderSampler(sampling_strategy=majority_mean, random_state=42)
    else:
        print("Wrong sampling strategy as argument: must be 'majority' or 'majority_mean'")
        exit(1)

    if args.shuffle == True:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=5)

    here = pathlib.Path(__file__).parent #path of this script
    csvdir = here.joinpath(args.path) #get the directory in which there are the files to analyze
    resultspath = here.joinpath(args.savepath) #path of the results folder

    #construct path for storing the result summary in txt
    resultOut = "classification_results.csv"
    result_path = resultspath.joinpath(resultOut)
    #construct path for saving png about classification performances
    measureType = "classification_results_image.png"
    result_image = resultspath.joinpath(measureType)

    rfe_file = csvdir.joinpath("RFEFeat.txt") #path of the file containing the features to extract
    rfe_feature_reader = FeatureReader(RFEFeatureReader(), str(rfe_file))
    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    experiments_scores = OrderedDict()
    dfs = list()

    for file_entry in tqdm(list(csvdir.iterdir())):
        if file_entry.is_file() and file_entry.suffix == '.csv':
            filepath = csvdir.joinpath(file_entry.name)
            print("Getting data from file ", file_entry.name)
            input_name = file_entry.name.split('.')[0]
            nodename = file_entry.name.split('_')[0].split('/')[-1]

            #get the dataframe considering only specific columns
            data = pd.read_csv(file_entry, usecols=selectionlist)
            dfs.append(data)

    #concatenate all the training dataframes in one dataframe
    train_df = pd.concat(dfs, ignore_index=True)
    X = train_df[featurelist].to_numpy()
    y = train_df['label'].to_numpy()
    labels = np.unique(y)

    #classifier model
    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)


    #array to memorize the scores
    scoreArray = np.zeros(shape=(5,9), dtype=np.float64, order='C')
    for fold, (train_index, test_index) in enumerate(kf.split(X), 0):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, y_train = sampler.fit_resample(X_train, y_train)
        X_test, y_test = sampler.fit_resample(X_test, y_test)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_weighted = f1_score(y_test, y_pred, average="weighted")
        test_all = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=1)
        scoreArray[fold, 0] = test_weighted
        scoreArray[fold, 1:] = test_all
        experiments_scores["Fold " + str(fold+1)] = list(scoreArray[fold, :])

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']
    overall_score = scoreArray[:,0].mean()
    
    key_score = [scoreArray[:,i].mean() for i in range(1, 9)]
    all_scores = [overall_score] + key_score

    experiments_scores['all'] = all_scores
    saveresults(experiments_scores, keys, result_path)
    plot_heatmap("F1-scores", experiments_scores, keys, result_image, True)

