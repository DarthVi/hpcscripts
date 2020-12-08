import random
from collections import OrderedDict
import os
os.environ["MODIN_ENGINE"] = "ray"
import time

import numpy as np
import modin.pandas as pd
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

from utils import plot_heatmap, str2bool, saveresults, updateboxplotsCSV, rawgencount
from FileFeatureReader.featurereaders import RFEFeatureReader, DTFeatureReader
from FileFeatureReader.featurereader import FeatureReader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path in which there are the files to analyze")
    parser.add_argument("-s", "--shuffle", type=str2bool, nargs='?', const=True, default=False, help="'yes' to enable shuffling, 'no' otherwise")
    parser.add_argument("-z", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-b", "--tmin", type=int, default=9, help="Lower bound that excludes nodes lower than this value from training")
    parser.add_argument("-c", "--tmax", type=int, default=24, help="Upper bound that excludes nodes higher than this value from training")
    parser.add_argument("-d", "--sumpath", type=str, default="./boxplots", help="Path in which CSVs summary for boxplots will be stored")
    parser.add_argument("-e", "--featfile", type=str, default="./RFEFile.txt", help="Path of the feature list file")
    parser.add_argument("-f", "--step", type=int, default=1, help="Sampling used to read the training CSVs, that is how many lines to skip before picking up an element")
    parser.add_argument("-g", "--balancetest", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the test set, 'no' otheriwse")
    parser.add_argument("-i", "--feattype", type=str, default="rfe", help="'rfe' to read from RFE file, 'dt' to read from dt-like file")
    parser.add_argument("-l", "--numfeat", type=int, default=100, help="Number of features to use when selecting them via DT file")
    parser.add_argument("-m", "--maxnumsample", type=int, default=16, help="Up to how many nodes to train the model on")
    parser.add_argument("-n", "--numiter", type=int, help="Iteration number of the current experiment")
    args = parser.parse_args()

    if args.step < 1:
        print("-f (--step) argument must be >= 1")
        exit(1)

    if args.feattype != "rfe" and args.feattype != 'dt':
        print("-i (--feattype) can be 'rfe' or 'dt'")
        exit(1)

    if args.feattype == 'dt' and args.numfeat < 1:
        print("-l (--numfeat) must be > 1")
        exit(1)

    if args.balancetest == False:
        print("Warning: class balancing will not be performed on the test set")

    if args.maxnumsample < 1:
        print("-m (--maxnumsample) must be >= 1")
        exit(1)

    csvdir = pathlib.Path(args.path) #get the directory in which there are the files to analyze
    sumpath = pathlib.Path(args.sumpath)
    featfile = pathlib.Path(args.featfile)


    csvlist = list(csvdir.glob("*.csv"))
    csvlist = sorted(csvlist, key=lambda x: int(x.stem.split('_')[0][1:]))
    #filter out edge nodes
    possible_trainnodes = list(filter(lambda x: args.tmin <= int(x.stem.split('_')[0][1:]) <= args.tmax, csvlist))
    #shuffle the list of CSVs randomly and then choose a random subsample of k computing nodes

    #get features used for training
    if args.feattype == 'rfe':
        rfe_feature_reader = FeatureReader(RFEFeatureReader(), featfile)
        featurelist = rfe_feature_reader.getFeats()
    else:
        dt_feature_reader = FeatureReader(DTFeatureReader(), featfile)
        featurelist = dt_feature_reader.getNFeats(args.numfeat)

    #add the label to the previous list
    selectionlist = featurelist + ['label']

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

    #train and test using up to args.maxnumsample training nodes
    for i in range(args.maxnumsample):
        np.random.seed(int(time.time()))
        random.seed(int(time.time()))
        #trick to return a new shuffled list (random.shuffle would modify the list in place)
        shuffled_training_pool = random.sample(possible_trainnodes, len(possible_trainnodes))
        num_train = i + 1
        trainnodes = random.sample(shuffled_training_pool, k=num_train)

        #get testing nodes subtracting the training nodes from the whole list of nodes
        testnodes = list(set(csvlist) - set(trainnodes))
        #sort testnodes in ascending order
        testnodes = sorted(testnodes, key=lambda x: int(x.stem.split('_')[0][1:]))


        lX = list()
        ly = list()

        for file_entry in tqdm(trainnodes):
            print("Getting data from file ", file_entry.name)
            #read how many lines there are in the CSV, excluding the header line
            numlines = rawgencount(file_entry) - 1
            #rows to skip, if sampling is 1, the list will be empty and no lines will be skipped
            skiplist = list(set(range(1, numlines + 1)) - set(range(1, numlines + 1, args.step)))
            data = pd.read_csv(file_entry, usecols=selectionlist, header=0, skiprows=skiplist)
            y = data['label'].to_numpy()
            X = data.drop(['label'], axis=1)
            #reorder column
            X = X[featurelist]
            X = X.to_numpy()
            del data
            #we do class balancing right here in the loop and not after, in order to save memory
            #random undersampling for class balancing
            rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=int(time.time()))
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
        clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=int(time.time()))

        #train the model and save it
        clf.fit(X, y)
        del X
        del y
        #dump(clf, modelfile)

        clsResults = OrderedDict()

        for file_entry in tqdm(testnodes):
            print("Testing on file ", file_entry.name)
            nodename = file_entry.name.split('_')[0].split('/')[-1]

            #get the dataframe considering only specific columns
            data = pd.read_csv(file_entry, usecols=selectionlist)
            y = data['label'].to_numpy()
            X = data.drop(['label'], axis=1)
            #reorder columns
            X = X[featurelist]
            X = X.to_numpy()
            del data
    #        labels = np.unique(y)

            #class balancing by random undersampling
            if args.balancetest == True:
                rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=int(time.time()))
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

        saveresults(clsResults, keys, "/tmp/classification_results.csv")

        #for each fault, update a CSV containing the fault scores for each node and the number of nodes used as training
        for col in keys:
            updateboxplotsCSV("/tmp/classification_results.csv", col, sumpath, num_train, args.numiter)
