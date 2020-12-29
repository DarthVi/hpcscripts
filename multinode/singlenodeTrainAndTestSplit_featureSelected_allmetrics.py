from collections import OrderedDict, Counter
import os
from copy import deepcopy
from pprint import pprint
import time

import numpy as np
import pandas as pd
import pathlib
from sklearn.ensemble import RandomForestClassifier
import sys
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import multilabel_confusion_matrix

from utils import plot_heatmap, plot_heatmap_light, str2bool, saveresults, updateboxplotsCSV, rawgencount, summaryboxplot, scoreboxplot
from eval_utils import get_f1_score, get_sensitivity, get_specificity, get_FP_rate, get_FN_rate
from FileFeatureReader.featurereaders import RFEFeatureReader, DTFeatureReader
from FileFeatureReader.featurereader import FeatureReader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path in which there are the files to analyze")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Seed used in the random generator for the selection of nodes")
    parser.add_argument("-v", "--savepath", type=str, default="./results", help="Path of the folder in which to save the experiment results")
    parser.add_argument("-a", "--annotations", type=str2bool, nargs='?', const=True, default=True, help="'yes' to annotate each cell of the heatmap, 'no' otherwise")
    parser.add_argument("-z", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-e", "--featfile", type=str, default="./RFEFile.txt", help="Path of the feature list file")
    parser.add_argument("-g", "--balancetest", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the test set, 'no' otheriwse")
    parser.add_argument("-n", "--balancetrain", type=str2bool, nargs='?', const=True, default=True, help="'yes' to balance the training nodes, 'no' otherwise")
    parser.add_argument("-i", "--feattype", type=str, default="rfe", help="'rfe' to read from RFE file, 'dt' to read from dt-like file")
    parser.add_argument("-l", "--numfeat", type=int, default=100, help="Number of features to use when selecting them via DT file")
    args = parser.parse_args()

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

    f_scores = OrderedDict()
    sensitivity_scores = OrderedDict()
    specificity_scores = OrderedDict()
    FP_scores = OrderedDict()
    FN_scores = OrderedDict()

    stime = time.time()
    for file_entry in tqdm(csvlist):
        print("Training and testing on file ", file_entry.name)
        nodename = file_entry.stem.split('_')[0].split('/')[-1]
        data = pd.read_csv(file_entry, usecols=selectionlist, header=0)
        #all the columns except the label one
        metricKeys = list(data.columns)[:-1]
        #select new data using only the most important columns
        features = data[featurelist]
        features = features.to_numpy()
        #select the labels
        labels = data['label']
        labels = labels.to_numpy()

        #60-40 train-test split
        numTrain = int(0.6*len(features))
        trainData = features[:numTrain]
        trainLbl = labels[:numTrain]
        testData = features[numTrain:]
        testLbl = labels[numTrain:]

        rus = RandomUnderSampler(sampling_strategy="majority", random_state=args.seed)

        clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=args.seed)

        if args.balancetrain == True:
            trainData, trainLbl = rus.fit_resample(trainData, trainLbl)
        if args.balancetest == True:
            testData, testLbl = rus.fit_resample(testData, testLbl)

        clf.fit(trainData, trainLbl)
        pred = clf.predict(testData)

        counters = Counter(testLbl.astype(int))
        #for debugging purposes
        #pred_counters = Counter(pred.astype(int))
        #print(counters)
        #print(pred_counters)
        mcm_ = multilabel_confusion_matrix(testLbl, pred)
        get_f1_score(testLbl, pred, f_scores, nodename)
        get_sensitivity(counters, mcm_, sensitivity_scores, nodename)
        get_specificity(counters, mcm_, specificity_scores, nodename)
        get_FP_rate(counters, mcm_, FP_scores, nodename)
        get_FN_rate(counters, mcm_, FN_scores, nodename)
    etime = time.time()

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

    print("Experiment took %f"%(etime-stime))
