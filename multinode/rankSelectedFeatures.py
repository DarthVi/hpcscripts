import re
import os
from collections import OrderedDict
import sys

import pathlib
import numpy as np
import pandas as pd
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from FileFeatureReader.featurereaders import RFEFeatureReader
from FileFeatureReader.featurereader import FeatureReader

if __name__ == '__main__':
    random.seed(42)

    if len(sys.argv) != 4:
        print("Please insert a file path to analyze as argument, a feature file and a save path")
        exit(1)

    input_file = pathlib.Path(str(sys.argv[1]))
    featfile = pathlib.Path(str(sys.argv[2]))
    savepath = pathlib.Path(str(sys.argv[3]))
    input_name = input_file.stem
    nodename = input_file.stem.split('_')[0].split('/')[-1]

    rfe_feature_reader = FeatureReader(RFEFeatureReader(), featfile)

    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    featureFile = savepath.joinpath(nodename + "_result_RFE_" + str(len(featurelist)) + "decTreeMostImportantRanking.txt")

    data = pd.read_csv(input_file, usecols=selectionlist)
    metricKeys = list(data.columns)[:-1]

    rus = RandomUnderSampler(sampling_strategy="majority", random_state=42)
    clf = DecisionTreeClassifier(random_state=42)

    #select the features (all the column except the label one)
    features = data[list(data.columns)[:-1]]
    features = features.to_numpy()
    #select the labels
    labels = data[list(data.columns)[-1]]
    labels = labels.to_numpy()

    #class balancing
    #features, labels = rus.fit_resample(features, labels)

    #60-40 train-test split
    # numTrain = int(0.6*len(features))
    # trainData = features[:numTrain]
    # trainLbl = labels[:numTrain]

    #trainData, trainLbl = rus.fit_resample(trainData, trainLbl)

    #clf.fit(trainData, trainLbl)

    features, labels = rus.fit_resample(features, labels)
    clf.fit(features, labels)

    with open(featureFile, 'w') as out:
        out.write('- Features ranking:\n')

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('- Features ranking:')
    with open(featureFile, 'a') as out:
       for idx in range(len(indices)):
           metric = indices[idx]
           print('---- %s: %s' % (metricKeys[metric], importances[metric]))
           out.write('---- %s: %s\n' % (metricKeys[metric], importances[metric]))