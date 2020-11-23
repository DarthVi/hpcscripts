import re
import os
from collections import OrderedDict
import sys

import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from FileFeatureReader.featurereaders import RFEFeatureReader
from FileFeatureReader.featurereader import FeatureReader
from tqdm import tqdm
import argparse
from utils import str2bool, plot_heatmap, saveresults

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--path", type=str, default="train/", help="Path in which there are the training files")
    parser.add_argument("-t", "--rfepath", type=str, default="", help="Path in which there is the list of most important features")
    parser.add_argument("-r", "--savepath", type=str, default="results/", help="Path in which to store the results")
    parser.add_argument("-s", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    parser.add_argument("-p", "--title", type=str, default="F1-scores", help="Title to give to the heatmap generated")
    parser.add_argument("-a", "--annotation", type=str2bool, nargs='?', const=True, default=True, help="Wether to annotate or not each cell of the heatmap")
    parser.add_argument("-b", "--seed", type=int, default=42, help="Seed to use for the RNG")
    args = parser.parse_args()

    np.random.seed(args.seed)

    csvpath = pathlib.Path(args.path)
    rfefilepath = pathlib.Path(args.rfepath)
    savepath = pathlib.Path(args.savepath)
    resultsfile = savepath.joinpath("classification_results.csv")
    summarypng = savepath.joinpath("summary.png")

    nodes = list(csvpath.glob("*.csv")) #get the node files
    nodes = sorted(nodes, key=lambda x: int(x.stem.split('_')[0][1:])) #order from N1, N2... up to N32 in ascending order

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

    #random undersampling for class balancing
    rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=args.seed)

    #classifier model
    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=args.seed)


    rfe_feature_reader = FeatureReader(RFEFeatureReader(), rfefilepath)

    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    clsResults = OrderedDict()


    for node in tqdm(nodes):
        print("Getting data from file ", node.name)
        nodename = node.name.split('_')[0].split('/')[-1]
        data = pd.read_csv(node, usecols=selectionlist)
        #select the features (all the column except the label one)
        features = data.drop(['label'], axis=1).to_numpy()
        #select the labels
        labels = data['label'].to_numpy()
        #60-40 train-test split
        numTrain = int(0.6*len(features))
        trainData = features[:numTrain]
        trainLbl = labels[:numTrain]
        testData = features[numTrain:]
        testLbl = labels[numTrain:]

        #class balancing for both training and testing
        trainData, trainLbl = rus.fit_resample(trainData, trainLbl)
        testData, testLbl = rus.fit_resample(testData, testLbl)
        clf.fit(trainData, trainLbl)
        del trainData
        del trainLbl
        pred = clf.predict(testData)
        del testData

        #list to store f1 values
        F = []

        print('- Classifier: %s' % clf.__class__.__name__)

        #calculate global overall F1-score with weighted average
        f1 = f1_score(testLbl, pred, average = 'weighted')
        F.append(f1)
        print('Overall score: %f.'%f1)

        #calculate F1-score for each class
        test_all = f1_score(testLbl, pred, average=None)
        F.extend(list(test_all))

        for i, f in enumerate(test_all):
            print('Fault: %d,  F1: %f.'%(i,f))

        clsResults[nodename] = F.copy()
        del testLbl
        del pred

    saveresults(clsResults, keys, resultsfile)
    plot_heatmap(args.title, clsResults, keys, summarypng, args.annotation)