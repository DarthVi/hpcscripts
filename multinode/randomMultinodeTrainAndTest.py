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
from joblib import dump
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
    parser.add_argument("-z", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    args = parser.parse_args()

    ranp.seed(42)
    random.seed(args.seed)

    here = pathlib.Path(__file__).parent #path of this script
    csvdir = here.joinpath(args.path) #get the directory in which there are the files to analyze
    resultsfile = here.joinpath(args.savepath).joinpath("classification_results.csv")
    trainfile = here.joinpath(args.savepath).joinpath("trainingnodes.csv")
    modelfile = here.joinpath(args.savepath).joinpath("model.joblib")
    summarypng = here.joinpath(args.savepath).joinpath("summary.png")

    experiments_scores = OrderedDict()

    csvlist = list(csvdir.glob("*.csv"))
    #shuffle the list of CSVs randomly and then choose a random subsample of 4 computing nodes
    random.shuffle(csvlist)
    trainnodes = random.sample(csvlist, k=args.sample)

    #get training nodes names from filepath
    trainnames = map(lambda x: str(x).split('/')[-1].split('_')[0], trainnodes)

    tf = pd.DataFrame(trainnames)
    #save nodes used as training
    tf.to_csv(trainfile)

    del tf

    #get testing nodes subtracting the training nodes from the whole list of nodes
    testnodes = list(set(csvlist) - set(trainnodes))
    #sort testnodes in ascending order
    testnodes = sorted(testnodes)
    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

    dfs = list()

    for file_entry in tqdm(trainnodes):
        print("Getting data from file ", file_entry.name)
        data = pd.read_csv(file_entry)
        dfs.append(data)
        del data

    #concatenate all the training dataframes in one dataframe
    train_df = pd.concat(dfs, ignore_index=True)
    del dfs

    #if shuffling is enabled
    if args.shuffle == True:
        #the idiomatic pandas way to shuffle is by using df.sample
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = train_df.drop(['label'], axis=1).to_numpy()
    y = train_df['label'].to_numpy()

    del train_df

    #classifier model
    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

    #random undersampling for class balancing
    rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=42)

    #class balancing
    X, y = rus.fit_resample(X, y)
    #train the model and save it
    clf.fit(X, y)
    del X
    del y
    dump(clf, modelfile)

    clsResults = OrderedDict()

    for file_entry in tqdm(testnodes):
        print("Testing on file ", file_entry.name)
        nodename = file_entry.name.split('_')[0].split('/')[-1]

        #get the dataframe considering only specific columns
        data = pd.read_csv(file_entry)
        X = data.drop(['label'], axis=1).to_numpy()
        y = data['label'].to_numpy()
        del data
        labels = np.unique(y)

        #class balancing by random undersampling
        X, y = rus.fit_resample(X, y)
        #predict labels for test set
        pred = clf.predict(X)
        del X

        #list to store f1 values
        F = []

        print('- Classifier: %s' % clf.__class__.__name__)

        #calculate global overall F1-score with weighted average
        f1 = f1_score(y, pred, average = 'weighted')
        F.append(f1)
        print('Overall score: %f.'%f1)

        #calculate score for each class by micro-averaging
        for l in np.unique(np.asarray(labels)):
            #select the correct labels
            lab_tmp = y[list(np.where(y == l)[0])]
            #select the corresponding predicted lables
            pred_tmp = pred[list(np.where(y == l)[0])]
            f1 = f1_score(lab_tmp, pred_tmp, average = 'micro', labels=labels, zero_division=1)
            F.append(f1)
            print('Fault: %d,  F1: %f.'%(l,f1))

        clsResults[nodename] = F.copy()
        del y

    saveresults(clsResults, keys, resultsfile)
    plot_heatmap(args.title, clsResults, keys, summarypng, args.annotations)
