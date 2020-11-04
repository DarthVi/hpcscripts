import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import pathlib
from numpy import random as ranp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

from utils import plot_heatmap, str2bool, saveresults

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="relative path in which there are the files to analyze")
    parser.add_argument("-s", "--shuffle", type=str2bool, nargs='?', const=True, default=False, help="'yes' to enable shuffling, 'no' otherwise")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Seed used in the random generator for the selection of nodes")
    parser.add_argument("-v", "--savepath", type=str, default="results/", help="relative path of the folder in which to save the experiment results")
    parser.add_argument("-a", "--annotations", type=str2bool, nargs='?', const=True, default=True, help="'yes' to annotate each cell of the heatmap, 'no' otherwise")
    parser.add_argument("-t", "--title", type=str, default="", help="Title to give to the heatmap generated")
    parser.add_argument("-k", "--sample", type=int, default=4, help="How many nodes to randomly take from the list of available nodes")
    args = parser.parse_args()

    ranp.seed(42)
    random.seed(args.seed)

    here = pathlib.Path(__file__).parent #path of this script
    csvdir = here.joinpath(args.path) #get the directory in which there are the files to analyze
    resultsfile = here.joinpath(args.savepath).joinpath("classification_results.csv")
    summarypng = here.joinpath(args.savepath).joinpath("summary.png")

    experiments_scores = OrderedDict()

    csvlist = list(csvdir.glob("*.csv"))
    #shuffle the list of CSVs randomly and then choose a random subsample of 4 computing nodes
    random.shuffle(csvlist)
    randomnodes = random.sample(csvlist, k=args.sample)

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

    for file_entry in tqdm(randomnodes):
        filepath = csvdir.joinpath(file_entry.name)
        print("Processing file " + file_entry.name)
        input_name = file_entry.name.split('.')[0]
        nodename = file_entry.name.split('_')[0].split('/')[-1]
        
        #get the dataframe considering only specific columns
        data = pd.read_csv(file_entry)
        X = data.drop(['label'], axis=1).to_numpy()
        y = data['label'].to_numpy()
        labels = np.unique(y)
        #random undersampler resample only the majority class
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)

        #classifier model
        clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

        if(args.shuffle == False):
            kf = KFold(n_splits=5)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

        #array to memorize the scores
        scoreArray = np.zeros(shape=(5,9), dtype=np.float64, order='C')
        for fold, (train_index, test_index) in enumerate(kf.split(X), 0):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, y_train = rus.fit_resample(X_train, y_train)
            X_test, y_test = rus.fit_resample(X_test, y_test)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            test_weighted = f1_score(y_test, y_pred, average="weighted")
            test_all = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=1)
            scoreArray[fold, 0] = test_weighted
            scoreArray[fold, 1:] = test_all

        overall_score = scoreArray[:,0].mean()
        
        key_score = [scoreArray[:,i].mean() for i in range(1, 9)]
        all_scores = [overall_score] + key_score

        #save overall score in dictionary with nodename as key
        experiments_scores[nodename] = all_scores

        del X_train
        del X_test
        del y_train
        del y_test
        del y_pred
        del data
        del X
        del y


    saveresults(experiments_scores, keys, resultsfile)
    plot_heatmap(args.title, experiments_scores, keys, summarypng, annotation=args.annotations)