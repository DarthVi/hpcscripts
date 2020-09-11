"""
2020-09-10 14:44

@author: Vito Vincenzo Covella
"""

import numpy as np
import pandas as pd
from numpy import random
from sklearn.ensemble import RandomForestClassifier
import sys
import matplotlib.pyplot as plt
import os
from joblib import dump
from FileFeatureReader.featurereaders import RFEFeatureReader, DTFeatureReader
from FileFeatureReader.featurereader import FeatureReader

if __name__ == '__main__':

    random.seed(42)

    if len(sys.argv) != 3:
        print("Please insert a file path to analyze as argument and a feature list file")
        exit(1)

    input_file = str(sys.argv[1])
    input_name = input_file.split('.')[0]
    nodename = input_file.split('_')[0].split('/')[-1]

    featurepath = str(sys.argv[2])
    rfe_feature_reader = FeatureReader(RFEFeatureReader(), featurepath)

    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    #get the dataframe considering only specific columns
    data = pd.read_csv(input_file, usecols=selectionlist)

    features = data[featurelist]
    labels = data['label']

    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

    clf.fit(features, labels)

    savefile = input_name + "_model.joblib"

    dump(clf, savefile)