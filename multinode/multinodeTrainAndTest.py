"""
2020-10-03 15:04
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
from joblib import dump
from FileFeatureReader.featurereaders import RFEFeatureReader
from FileFeatureReader.featurereader import FeatureReader
from utils import str2bool, plot_heatmap, saveresults

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--trainpath", type=str, default="train/", help="Path in which there are the training files")
    parser.add_argument("-t", "--testpath", type=str, default="test/", help="Path in which there are the test files")
    parser.add_argument("-r", "--savepath", type=str, default="results/", help="Path in which to store the results")
    parser.add_argument("-s", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-p", "--title", type=str, default="F1-scores", help="Title to give to the heatmap generated")
    parser.add_argument("-a", "--annotation", type=str2bool, nargs='?', const=True, default=True, help="Wether to annotate or not each cell of the heatmap")
    args = parser.parse_args()

    random.seed(42)

    here = pathlib.Path(__file__).parent #path of this script
    trainpath = here.joinpath(args.trainpath) #path in which there are the training files
    testpath = here.joinpath(args.testpath) #path in which there are the testing files
    resultspath = here.joinpath(args.savepath) #path of the results folder
    rfe_file = trainpath.joinpath("RFEFeat.txt")

    rfe_feature_reader = FeatureReader(RFEFeatureReader(), rfe_file)

    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    train_nodenames = list()
    dfs = list()

    #loop over all the CSV containing training data, in order to extract it
    for file_entry in tqdm(list(trainpath.iterdir())):
        if file_entry.is_file() and file_entry.suffix == '.csv':
            filepath = trainpath.joinpath(file_entry.name)
            print("Getting data from file ", file_entry.name)
            input_name = file_entry.name.split('.')[0]
            nodename = file_entry.name.split('_')[0].split('/')[-1]
            train_nodenames.append(nodename)

            #get the dataframe considering only specific columns
            data = pd.read_csv(file_entry, usecols=selectionlist)
            dfs.append(data)

    #concatenate all the training dataframes in one dataframe
    train_df = pd.concat(dfs, ignore_index=True)
    X = train_df[featurelist].to_numpy()
    y = train_df['label'].to_numpy()

    #classifier model
    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

    #random undersampling for class balancing
    rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=42)

    #train the model
    X, y = rus.fit_resample(X, y)
    clf.fit(X, y)

    #save the model
    model_savefile = '_'.join(train_nodenames) + "_model.joblib"
    model_savepath = resultspath.joinpath(model_savefile)
    dump(clf, model_savepath)

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']
    clsResults = OrderedDict()

    for file_entry in tqdm(list(testpath.iterdir())):
        if file_entry.is_file() and file_entry.suffix == '.csv':
            filepath = testpath.joinpath(file_entry.name)
            print("Testing on file ", file_entry.name)
            input_name = file_entry.name.split('.')[0]
            nodename = file_entry.name.split('_')[0].split('/')[-1]

            #get the dataframe considering only specific columns
            data = pd.read_csv(file_entry, usecols=selectionlist)
            X = data[featurelist].to_numpy()
            y = data['label']
            y = y.to_numpy()

            #class balancing by random undersampling
            X, y = rus.fit_resample(X, y)
            labels = np.unique(np.asarray(y))

            pred = clf.predict(X)

            #list to store f1 values
            F = []

            print('- Classifier: %s' % clf.__class__.__name__)

            #calculate global overall F1-score with weighted average
            f1 = f1_score(y, pred, average = 'weighted')
            F.append(f1)
            print('Overall score: %f.'%f1)

            #calculate F1-score for each class
            test_all = f1_score(y, pred, average=None, labels=labels)
            F.extend(list(test_all))

            for i, f in enumerate(test_all):
                print('Fault: %d,  F1: %f.'%(i,f))

            clsResults[nodename] = F.copy()

    #construct path for storing the result summary in txt
    resultOut = "classification_results.csv"
    result_path = resultspath.joinpath(resultOut)
    #construct path for saving png about classification performances
    measureType = "classification_results_image.png"
    result_image = resultspath.joinpath(measureType)

    saveresults(clsResults, keys, result_path)
    plot_heatmap(args.title, clsResults, keys, result_image, args.annotation)