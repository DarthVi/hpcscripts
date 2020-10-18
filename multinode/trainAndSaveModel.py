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
from utils import majority_mean, minority_mean
from imblearn.under_sampling import RandomUnderSampler

if __name__ == '__main__':

    random.seed(42)

    if len(sys.argv) != 4:
        print("Please insert a file path to analyze as argument, a feature list file and 0 for mean undersampling or 1 for majority undersampling")
        exit(1)

    input_file = str(sys.argv[1])
    input_name = input_file.split('.')[0]
    nodename = input_file.split('_')[0].split('/')[-1]

    featurepath = str(sys.argv[2])
    rfe_feature_reader = FeatureReader(RFEFeatureReader(), featurepath)

    sampling_strategy = int(sys.argv[3])

    if sampling_strategy == 0:
        sampler = RandomUnderSampler(sampling_strategy=majority_mean, random_state=42)
    elif sampling_strategy == 1:
        sampler = RandomUnderSampler(sampling_strategy="majority", random_state=42)
    else:
        print("Invalid sampling strategy passed as command line argument")
        exit(1)

    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    #get the dataframe considering only specific columns
    data = pd.read_csv(input_file, usecols=selectionlist)

    features = data[featurelist]
    labels = data['label']

    #resampling for class balancing
    X_res, y_res = sampler.fit_resample(features, labels)

    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

    clf.fit(X_res, y_res)

    savefile = input_name + "_model.joblib"

    dump(clf, savefile)