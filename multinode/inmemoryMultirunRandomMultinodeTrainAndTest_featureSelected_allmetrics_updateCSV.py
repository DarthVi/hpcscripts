import random
from collections import OrderedDict, Counter
import os
import time

import numpy as np
import pandas as pd
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import multilabel_confusion_matrix

from utils import str2bool, appendresults
from eval_utils import get_f1_score, get_sensitivity, get_specificity, get_FP_rate, get_FN_rate
from FileFeatureReader.featurereaders import RFEFeatureReader, DTFeatureReader
from FileFeatureReader.featurereader import FeatureReader

def load_all_CSVs(pathlist, cols=None):
    '''Loads all CSVs in memory, returning a dictionary of dataframes'''
    dic = OrderedDict()
    for file in tqdm(pathlist):
        print("Loading in memory file: ", file.name)
        nodename = file.stem.split('_')[0]
        dic[nodename] = pd.read_csv(file, header=0, usecols=cols)
    return dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path in which there are the files to analyze")
    parser.add_argument("-s", "--shuffle", type=str2bool, nargs='?', const=True, default=False, help="'yes' to enable shuffling, 'no' otherwise")
    parser.add_argument("-z", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Seed used in the random generator for the selection of nodes")
    parser.add_argument("-b", "--tmin", type=int, default=9, help="Lower bound that excludes nodes lower than this value from training")
    parser.add_argument("-c", "--tmax", type=int, default=24, help="Upper bound that excludes nodes higher than this value from training")
    parser.add_argument("-d", "--sumpath", type=str, default="./boxplots", help="Path in which CSVs summary for boxplots will be stored")
    parser.add_argument("-e", "--featfile", type=str, default="./RFEFile.txt", help="Path of the feature list file")
    parser.add_argument("-f", "--step", type=int, default=1, help="Sampling used to read the training CSVs, that is how many lines to skip before picking up an element")
    parser.add_argument("-g", "--balancetest", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the test set, 'no' otheriwse")
    parser.add_argument("-o", "--balancetrain", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the training nodes, 'no' otherwise")
    parser.add_argument("-i", "--feattype", type=str, default="rfe", help="'rfe' to read from RFE file, 'dt' to read from dt-like file")
    parser.add_argument("-l", "--numfeat", type=int, default=100, help="Number of features to use when selecting them via DT file")
    parser.add_argument("-k", "--numsample", type=int, default=5, help="Up to how many nodes to train the model on")
    parser.add_argument("-n", "--numiter", type=int, default=10, help="Number of runs for this experiment")
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

    if args.balancetrain == False:
        print("Warning: class balancing will not be performed on the train set")

    if args.numsample < 1:
        print("-m (--numsample) must be >= 1")
        exit(1)

    if args.numiter < 1:
        print("-n (--numiter) must be >= 1")
        exit(1)

    csvdir = pathlib.Path(args.path) #get the directory in which there are the files to analyze
    sumpath = pathlib.Path(args.sumpath)
    featfile = pathlib.Path(args.featfile)

    sumpath.joinpath("fscore").mkdir(parents=True, exist_ok=True)
    sumpath.joinpath("sensitivity").mkdir(parents=True, exist_ok=True)
    sumpath.joinpath("specificity").mkdir(parents=True, exist_ok=True)
    sumpath.joinpath("FP_rate").mkdir(parents=True, exist_ok=True)
    sumpath.joinpath("FN_rate").mkdir(parents=True, exist_ok=True)


    csvlist = list(csvdir.glob("*.csv"))
    csvlist = sorted(csvlist, key=lambda x: int(x.stem.split('_')[0][1:]))

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

    #build dataframe dictionary loading everything into memory upfront
    df_dic = load_all_CSVs(csvlist, selectionlist)
    #get dataframe dictionary key list
    nodenamelist = list(df_dic.keys())
    nodenamelist = sorted(nodenamelist, key=lambda x: int(x[1:]))
    
    #filter out edge nodes
    possible_trainnodes = list(filter(lambda x: args.tmin <= int(x[1:]) <= args.tmax, nodenamelist))

    np.random.seed(int(time.time()))
    random.seed(int(time.time()))

    #repeat experiments args.numiter times
    for i in range(args.numiter):
        num_iter = i + 1
        print("Iteration number: ", num_iter)
        #train and test using up to args.maxnumsample training nodes
        stime = time.time()
        num_train = args.numsample
        print("Iteration %d, training on %d nodes"%(num_iter, num_train))
        #trick to return a new shuffled list (random.shuffle would modify the list in place)
        #shuffled_training_pool = random.sample(possible_trainnodes, len(possible_trainnodes))
        random.shuffle(possible_trainnodes)
        #take randomly k training nodes
        #trainnodes = random.sample(shuffled_training_pool, k=num_train)
        trainnodes = random.sample(possible_trainnodes, k=num_train)

        #get testing nodes subtracting the training nodes from the whole list of nodes
        testnodes = list(set(nodenamelist) - set(trainnodes))
        #sort testnodes in ascending order
        testnodes = sorted(testnodes, key=lambda x: int(x[1:]))


        lX = list()
        ly = list()

        for node in tqdm(trainnodes):
            print("Getting data from node ", node)
            data = df_dic[node].copy()
            #if subsampling is enabled, subsample
            if args.step > 1:
                data = data.iloc[::args.step, :]
            y = data['label'].to_numpy()
            X = data.drop(['label'], axis=1)
            #reorder column
            X = X[featurelist]
            X = X.to_numpy()
            del data
            #we do class balancing right here in the loop and not after, in order to save memory
            #random undersampling for class balancing
            if args.balancetrain == True:
                rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=args.seed)
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

        #train the model and save it
        clf.fit(X, y)
        del X
        del y
        #dump(clf, modelfile)

        f_scores = OrderedDict()
        sensitivity_scores = OrderedDict()
        specificity_scores = OrderedDict()
        FP_scores = OrderedDict()
        FN_scores = OrderedDict()

        for node in tqdm(testnodes):
            print("Testing on node ", node)
            nodename = node

            data = df_dic[node].copy()
            y = data['label'].to_numpy()
            X = data.drop(['label'], axis=1)
            #reorder columns
            X = X[featurelist]
            X = X.to_numpy()
            del data
    #        labels = np.unique(y)

            #class balancing by random undersampling
            if args.balancetest == True:
                rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=args.seed)
                X, y = rus.fit_resample(X, y)
            #predict labels for test set
            pred = clf.predict(X)
            del X

            counters = Counter(y.astype(int))
            mcm_ = multilabel_confusion_matrix(y, pred)
            get_f1_score(y, pred, f_scores, nodename)
            get_sensitivity(counters, mcm_, sensitivity_scores, nodename)
            get_specificity(counters, mcm_, specificity_scores, nodename)
            get_FP_rate(counters, mcm_, FP_scores, nodename)
            get_FN_rate(counters, mcm_, FN_scores, nodename)

            del y

        appendresults(f_scores, keys, sumpath.joinpath("fscore/classification_results.csv"))
        appendresults(sensitivity_scores, keys, sumpath.joinpath("sensitivity/classification_results.csv"))
        appendresults(specificity_scores, keys, sumpath.joinpath("specificity/classification_results.csv"))
        appendresults(FP_scores, keys, sumpath.joinpath("FP_rate/classification_results.csv"))
        appendresults(FN_scores, keys, sumpath.joinpath("FN_rate/classification_results.csv"))

        etime = time.time()
        print("One iteration took: ", etime - stime)
