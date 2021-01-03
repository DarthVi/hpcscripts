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

from utils import plot_heatmap, plot_heatmap_light, str2bool, saveresults, updateboxplotsCSV, summaryboxplot, scoreboxplot, grouped_scoreboxplot
from eval_utils import get_f1_score, get_sensitivity, get_specificity, get_FP_rate, get_FN_rate
from FileFeatureReader.featurereaders import RFEFeatureReader, DTFeatureReader
from FileFeatureReader.featurereader import FeatureReader

def readhalvesfile(path):
    with open(path) as infile:
        s = list(map(str.strip, infile))[0]
    return s

def get_complementary_node(trainlist):
    complementary = list()
    traindic = dict(trainlist)
    #remove items that occurr more than once
    remove_double = lambda data, c: [x for x in set(data) if data.count(x) <= c]
    no_dup = remove_double([x[0] for x in trainlist], 1)
    if no_dup:
        for n in no_dup:
            if traindic[n] == 1:
                complementary.append((n, 2))
            else:
                complementary.append((n, 1))
    return complementary

def getdataframe(tup, columns, printstm):
    print(printstm)
    d = pd.read_csv(tup[0], usecols=columns, header=0)
    name = tup[0].name.split('_')[0].split('/')[-1]
    if tup[1] != 0:
        half_len = int(len(d)/2)
        name = tup[0].name.split('_')[0].split('/')[-1] + '_' + str(tup[1])
        #if second element of the tuple is 1, return first half
        if tup[1] == 1:
            print("Getting only first half")
            return (name, d.iloc[:half_len, :])
        #otherwise return second half
        else:
            print("Getting only second half")
            return (name, d.iloc[half_len:, :])
    #if tup[1] == 0
    return (name, d)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path in which there are the files to analyze")
    parser.add_argument("-s", "--shuffle", type=str2bool, nargs='?', const=True, default=False, help="'yes' to enable shuffling, 'no' otherwise")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Seed used in the random generator for the selection of nodes")
    parser.add_argument("-v", "--savepath", type=str, default="./results", help="Path of the folder in which to save the experiment results")
    parser.add_argument("-a", "--annotations", type=str2bool, nargs='?', const=True, default=True, help="'yes' to annotate each cell of the heatmap, 'no' otherwise")
    parser.add_argument("-z", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-e", "--featfile", type=str, default="./RFEFile.txt", help="Path of the feature list file")
    parser.add_argument("-f", "--step", type=int, default=1, help="Sampling used to read the training CSVs, that is how many lines to skip before picking up an element")
    parser.add_argument("-g", "--balancetest", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the test set, 'no' otheriwse")
    parser.add_argument("-n", "--balancetrain", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the training nodes, 'no' otherwise")
    parser.add_argument("-i", "--feattype", type=str, default="rfe", help="'rfe' to read from RFE file, 'dt' to read from dt-like file")
    parser.add_argument("-l", "--numfeat", type=int, default=100, help="Number of features to use when selecting them via DT file")
    parser.add_argument("-m", "--clustering", type=str, default="./clustering", help="Path of the folder containing the results of clustering")
    parser.add_argument("-o", "--baselinepath", type=str, default="./baseline", help="Path of the folder containing the baseline results")
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
    clustpath = pathlib.Path(args.clustering)
    candidatepath = clustpath.joinpath("candidate_nodes.csv")
    shalvespath = clustpath.joinpath("separate_halves.txt")

    baselinepath = pathlib.Path(args.baselinepath)
    baselinefscore = baselinepath.joinpath("fscore/classification_results.csv")
    baselinesensitivity = baselinepath.joinpath("sensitivity/classification_results.csv")
    baselinespecificity = baselinepath.joinpath("specificity/classification_results.csv")
    baselinefpscore = baselinepath.joinpath("FP_score/classification_results.csv")
    baselinefnscore = baselinepath.joinpath("FN_score/classification_results.csv")

    csvlist = list(csvdir.glob("*.csv"))
    csvlist = sorted(csvlist, key=lambda x: int(x.stem.split('_')[0][1:]))

    #get filenames suffix
    suffix = csvlist[0].name.split('_', maxsplit=1)[1]

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

    #get the training nodes and testing nodes
    halves = readhalvesfile(shalvespath)

    tf = pd.read_csv(candidatepath, header=0, index_col=0)
    trainnodesname = tf["nodename"].values.tolist()

    if halves == "yes":
        print("halves == yes")
        nodes = [(x.split('_')[0], int(x.split('_')[1])) for x in trainnodesname]
        print(nodes)
        trainnodes = list(map(lambda x: (csvdir.joinpath(x[0] + '_' + suffix), x[1]), nodes))
        #pprint(trainnodes)
        testnodes = list(set(csvlist) - set(list(map(lambda x: x[0], trainnodes))))
        testnodes = list(map(lambda x: (x, 0), testnodes))
        #pprint(testnodes)
        compl = get_complementary_node(nodes)
        #print(compl)
        testnodes.extend([(csvdir.joinpath(x[0] + '_' + suffix), x[1]) for x in compl])
        #pprint(testnodes)
        testnodes = sorted(testnodes, key=lambda x: int(x[0].stem.split('_')[0][1:]))
        #pprint(testnodes)
    else:
        trainnodes = list(map(lambda x: csvdir.joinpath(x + '_' + suffix), trainnodesname))
        testnodes = list(set(csvlist) - set(trainnodes))
        testnodes = sorted(testnodes, key=lambda x: int(x.stem.split('_')[0][1:]))
        trainnodes = [(x, 0) for x in trainnodes]
        testnodes = [(x, 0) for x in testnodes]
    del tf


    stime = time.time()
    spath = savepath
    spath.mkdir(parents=True, exist_ok=True)

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

    num_train = len(trainnodes)
    print("Experiment with %d training nodes"%(num_train))
    print(trainnodesname)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #random undersampling for class balancing
    rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=args.seed)

    lX = list()
    ly = list()

    for train_tup in tqdm(trainnodes):
        pstm = "Getting data from file " + train_tup[0].name
        nodename, data = getdataframe(train_tup, selectionlist, pstm)
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


    #labels = np.unique(y)


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

    for test_tup in tqdm(testnodes):
        pstm = "Testing on file " + test_tup[0].name
        nodename, data = getdataframe(test_tup, selectionlist, pstm)
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
        get_f1_score(y, pred, f_scores, nodename)
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

    plot_heatmap_light("F1-scores", f_scores, keys, fscore_path.joinpath("summary_light.png"), args.annotations)
    plot_heatmap_light("Sensitivity", sensitivity_scores, keys, sensitivity_path.joinpath("summary_light.png"), args.annotations)
    plot_heatmap_light("Specificity", specificity_scores, keys, specificity_path.joinpath("summary_light.png"), args.annotations)
    plot_heatmap_light("False Positive rate", FP_scores, keys, FP_path.joinpath("summary_light.png"), args.annotations)
    plot_heatmap_light("False Negative rate", FN_scores, keys, FN_path.joinpath("summary_light.png"), args.annotations)

    scoreboxplot(fscore_path.joinpath("classification_results.csv"), fscore_path.joinpath("boxplot.png"), "F1-scores")
    scoreboxplot(sensitivity_path.joinpath("classification_results.csv"), sensitivity_path.joinpath("boxplot.png"), "Sensitivity")
    scoreboxplot(specificity_path.joinpath("classification_results.csv"), specificity_path.joinpath("boxplot.png"), "Specificity")
    scoreboxplot(FP_path.joinpath("classification_results.csv"), FP_path.joinpath("boxplot.png"), "False Positive rate")
    scoreboxplot(FN_path.joinpath("classification_results.csv"), FN_path.joinpath("boxplot.png"), "False Negative rate")

    grouped_scoreboxplot(fscore_path.joinpath("classification_results.csv"), fscore_path.joinpath("boxplot_withbaseline.png"), baselinefscore, "F1-scores")
    grouped_scoreboxplot(sensitivity_path.joinpath("classification_results.csv"), sensitivity_path.joinpath("boxplot_withbaseline.png"), baselinesensitivity, "Sensitivity")
    grouped_scoreboxplot(specificity_path.joinpath("classification_results.csv"), specificity_path.joinpath("boxplot_withbaseline.png"), baselinespecificity, "Specificity")
    grouped_scoreboxplot(FP_path.joinpath("classification_results.csv"), FP_path.joinpath("boxplot_withbaseline.png"), baselinefpscore, "False Positive rate")
    grouped_scoreboxplot(FN_path.joinpath("classification_results.csv"), FN_path.joinpath("boxplot_withbaseline.png"), baselinefnscore, "False Negative rate")

    etime = time.time()
    print("Experiment took %f"%(etime-stime))

    
