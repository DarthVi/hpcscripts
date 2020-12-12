import random
from collections import OrderedDict, Counter
import os
from copy import deepcopy
from pprint import pprint
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
from joblib import dump
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import multilabel_confusion_matrix

from utils import plot_heatmap, str2bool, saveresults, updateboxplotsCSV, rawgencount, summaryboxplot
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
    parser.add_argument("-r", "--seed", type=int, default=42, help="Seed used in the random generator for the selection of nodes")
    parser.add_argument("-v", "--savepath", type=str, default="./results", help="Path of the folder in which to save the experiment results")
    parser.add_argument("-a", "--annotations", type=str2bool, nargs='?', const=True, default=True, help="'yes' to annotate each cell of the heatmap, 'no' otherwise")
    parser.add_argument("-z", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-b", "--tmin", type=int, default=9, help="Lower bound that excludes nodes lower than this value from training")
    parser.add_argument("-c", "--tmax", type=int, default=24, help="Upper bound that excludes nodes higher than this value from training")
    parser.add_argument("-e", "--featfile", type=str, default="./RFEFile.txt", help="Path of the feature list file")
    parser.add_argument("-f", "--step", type=int, default=1, help="Sampling used to read the training CSVs, that is how many lines to skip before picking up an element")
    parser.add_argument("-g", "--balancetest", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the test set, 'no' otheriwse")
    parser.add_argument("-n", "--balancetrain", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the training nodes, 'no' otherwise")
    parser.add_argument("-i", "--feattype", type=str, default="rfe", help="'rfe' to read from RFE file, 'dt' to read from dt-like file")
    parser.add_argument("-l", "--numfeat", type=int, default=100, help="Number of features to use when selecting them via DT file")
    parser.add_argument("-m", "--maxnumsample", type=int, default=16, help="Up to how many nodes to train the model on")
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
        print("Warning: class balancing will not be performed on the training nodes")

    np.random.seed(args.seed)
    random.seed(args.seed)

    csvdir = pathlib.Path(args.path) #get the directory in which there are the files to analyze  
    featfile = pathlib.Path(args.featfile)
    savepath = pathlib.Path(args.savepath)

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

    for i in range(args.maxnumsample):
        stime = time.time()
        spath = savepath.joinpath("RN" + str(i+1) + "f")
        spath.mkdir(parents=True, exist_ok=True)
        trainfile = spath.joinpath("trainingnodes.csv")

        #create paths
        fscore_path = spath.joinpath("fscore")
        sensitivity_path = spath.joinpath("sensitivity")
        specificity_path = spath.joinpath("specificity")
        FP_path = spath.joinpath("FP_score")
        FN_path = spath.joinpath("FN_score")

        fscore_path.mkdir(parents=True, exist_ok=True)
        sensitivity_path.mkdir(parents=True, exist_ok=True)
        specificity_path.mkdir(parents=True, exist_ok=True)
        FP_path.mkdir(parents=True, exist_ok=True)
        FN_path.mkdir(parents=True, exist_ok=True)

        savepath.joinpath("boxplotcsv/fscore").mkdir(parents=True, exist_ok=True)
        savepath.joinpath("boxplotcsv/sensitivity").mkdir(parents=True, exist_ok=True)
        savepath.joinpath("boxplotcsv/specificity").mkdir(parents=True, exist_ok=True)
        savepath.joinpath("boxplotcsv/FP_rate").mkdir(parents=True, exist_ok=True)
        savepath.joinpath("boxplotcsv/FN_rate").mkdir(parents=True, exist_ok=True)

        num_train = i + 1
        print("Experiment with %d training nodes"%(num_train))
        np.random.seed(args.seed)
        random.seed(args.seed)

        shuffled_training_pool = deepcopy(possible_trainnodes)
        random.shuffle(shuffled_training_pool)

        #take randomly k training nodes
        trainnodes = random.sample(shuffled_training_pool, k=num_train)

        tf = pd.DataFrame(trainnodes)
        #save nodes used as training
        tf.to_csv(trainfile)
        del tf

        #get testing nodes subtracting the training nodes from the whole list of nodes
        testnodes = list(set(nodenamelist) - set(trainnodes))
        #sort testnodes in ascending order
        testnodes = sorted(testnodes, key=lambda x: int(x[1:]))

        #random undersampling for class balancing
        rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=args.seed)

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
            if args.balancetrain == True:
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
                X, y = rus.fit_resample(X, y)
            #predict labels for test set
            pred = clf.predict(X)
            del X

            counters = Counter(y.astype(int))
            mcm_ = multilabel_confusion_matrix(y, pred)
            get_f1_score(y, pred, f_scores, nodename, labels)
            get_sensitivity(counters, mcm_, sensitivity_scores, nodename)
            get_specificity(counters, mcm_, specificity_scores, nodename)
            get_FP_rate(counters, mcm_, FP_scores, nodename)
            get_FN_rate(counters, mcm_, FN_scores, nodename)

            del y

        saveresults(f_scores, keys, fscore_path.joinpath("classification_results.csv"))
        saveresults(sensitivity_scores, keys, sensitivity_path.joinpath("classification_results.csv"))
        saveresults(specificity_scores, keys, specificity_path.joinpath("classification_results.csv"))
        saveresults(FP_scores, keys, FP_path.joinpath("classification_results.csv"))
        saveresults(FN_scores, keys, FN_path.joinpath("classification_results.csv"))
        
        plot_heatmap("F1-scores", f_scores, keys, fscore_path.joinpath("summary.png"), args.annotations)
        plot_heatmap("Sensitivity", sensitivity_scores, keys, sensitivity_path.joinpath("summary.png"), args.annotations)
        plot_heatmap("Specificity", specificity_scores, keys, specificity_path.joinpath("summary.png"), args.annotations)
        plot_heatmap("False Positive rate", FP_scores, keys, FP_path.joinpath("summary.png"), args.annotations)
        plot_heatmap("False Negative rate", FN_scores, keys, FN_path.joinpath("summary.png"), args.annotations)

        for col in keys:
            updateboxplotsCSV(fscore_path.joinpath("classification_results.csv"), col, savepath.joinpath("boxplotcsv/fscore"), num_train)
            updateboxplotsCSV(sensitivity_path.joinpath("classification_results.csv"), col, savepath.joinpath("boxplotcsv/sensitivity"), num_train)
            updateboxplotsCSV(specificity_path.joinpath("classification_results.csv"), col, savepath.joinpath("boxplotcsv/specificity"), num_train)
            updateboxplotsCSV(FP_path.joinpath("classification_results.csv"), col, savepath.joinpath("boxplotcsv/FP_rate"), num_train)
            updateboxplotsCSV(FN_path.joinpath("classification_results.csv"), col, savepath.joinpath("boxplotcsv/FN_rate"), num_train)

        etime = time.time()
        print("Training on %d nodes took %f seconds"%(num_train, etime-stime))

    for k in keys:
        summaryboxplot(savepath.joinpath("boxplotcsv/fscore"), k, savepath.joinpath("boxplotcsv/fscore"), 'num_train', k + ' F1-scores')
        summaryboxplot(savepath.joinpath("boxplotcsv/sensitivity"), k, savepath.joinpath("boxplotcsv/sensitivity"), 'num_train', k + ' sensitivity')
        summaryboxplot(savepath.joinpath("boxplotcsv/specificity"), k, savepath.joinpath("boxplotcsv/specificity"), 'num_train', k + ' specificity')
        summaryboxplot(savepath.joinpath("boxplotcsv/FP_rate"), k, savepath.joinpath("boxplotcsv/FP_rate"), 'num_train', k + ' FP_rate')
        summaryboxplot(savepath.joinpath("boxplotcsv/FN_rate"), k, savepath.joinpath("boxplotcsv/FN_rate"), 'num_train', k + ' FN_rate')
