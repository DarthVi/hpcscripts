import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import pathlib
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
    parser.add_argument("-g", "--selectionpath", type=str, help="relative path of the folder containing the CSV of training node to select")
    parser.add_argument("-s", "--shuffle", type=str2bool, nargs='?', const=True, default=False, help="'yes' to enable shuffling, 'no' otherwise")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Seed used in the random generator for the selection of nodes")
    parser.add_argument("-v", "--savepath", type=str, default="results/", help="relative path of the folder in which to save the experiment results")
    parser.add_argument("-a", "--annotations", type=str2bool, nargs='?', const=True, default=True, help="'yes' to annotate each cell of the heatmap, 'no' otherwise")
    parser.add_argument("-t", "--title", type=str, default="", help="Title to give to the heatmap generated")
    parser.add_argument("-z", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-f", "--suffix", type=str, default="_11f_60s_1step_mode", help="Suffix used for feature vectors names")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    here = pathlib.Path(__file__).parent #path of this script
    csvdir = here.joinpath(args.path) #get the directory in which there are the files to analyze
    selectionfile = here.joinpath(args.selectionpath).joinpath("nodes_selected.csv")
    resultsfile = here.joinpath(args.savepath).joinpath("classification_results.csv")
    trainfile = here.joinpath(args.savepath).joinpath("trainingnodes.csv")
    modelfile = here.joinpath(args.savepath).joinpath("model.joblib")
    summarypng = here.joinpath(args.savepath).joinpath("summary.png")

    experiments_scores = OrderedDict()

    #list of ALL the CSVs
    csvlist = list(csvdir.glob("*.csv"))

    #get the training nodes name from the related CSV
    tf = pd.read_csv(selectionfile, header=0, index_col=0)
    trainames = [x[1] for x in tf['0'].iteritems()]
    #build training feature vectors names
    trainfvectors = map(lambda x: x + args.suffix + ".csv", trainames)

    #Path of the training feature vectors
    trainnodes = [csvdir.joinpath(x) for x in trainfvectors]


    stf = pd.DataFrame(trainames)
    #save nodes used as training
    stf.to_csv(trainfile)

    del stf
    del tf

    #get testing nodes subtracting the training nodes from the whole list of nodes
    testnodes = list(set(csvlist) - set(trainnodes))
    #sort testnodes in ascending order
    testnodes = sorted(testnodes, key=lambda x: int(x.stem.split('_')[0][1:]))
    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

    #random undersampling for class balancing
    rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=args.seed)

    lX = list()
    ly = list()

    columnlist = list()

    for file_entry in tqdm(trainnodes):
        print("Getting data from file ", file_entry.name)
        data = pd.read_csv(file_entry)

        #initialize columnlist to use to reorder columns if this is empty
        if not columnlist:
            columnlist = list(data.columns)
            columnlist.remove('label')

        y = data['label'].to_numpy()
        X = data.drop(['label'], axis=1)
        X = X[columnlist]
        X = X.to_numpy()
        del data
        #we do class balancing right here in the loop and not after, in order to save memory
        X, y = rus.fit_resample(X, y)
        lX.append(X)
        ly.append(y)
        del X
        del y

    #concatenate all the training data
    X = np.concatenate(lX, axis=0)
    y = np.concatenate(ly, axis=0)
    del lX
    del ly

    #if shuffling is enabled
    if args.shuffle == True:
        #get current RNG state
        rng_state = np.random.get_state()
        np.random.shuffle(X)
        #reset RNG state to the previous state in order to shuffle y in the same way X has been shuffled
        np.random.set_state(rng_state)
        np.random.shuffle(y)


    labels = np.unique(y)

    #classifier model
    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=args.seed)

    #random undersampling for class balancing
    rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=args.seed)

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
        y = data['label'].to_numpy()

        X = data.drop(['label'], axis=1)
        X = X[columnlist]
        X = X.to_numpy()
        del data
#        labels = np.unique(y)

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

        #calculate F1-score for each class
        test_all = f1_score(y, pred, average=None, labels=labels)
        F.extend(list(test_all))

        for i, f in enumerate(test_all):
            print('Fault: %d,  F1: %f.'%(i,f))

        clsResults[nodename] = F.copy()
        del y

    saveresults(clsResults, keys, resultsfile)
    plot_heatmap(args.title, clsResults, keys, summarypng, args.annotations)
