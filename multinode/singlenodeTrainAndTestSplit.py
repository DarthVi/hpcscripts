import re
import os
from collections import OrderedDict

import pathlib
import numpy as np
import pandas as pd
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

from utils import plot_heatmap

def truncate(num,decimal_places):
    dp = str(decimal_places)
    return float(re.sub(r'^(\d+\.\d{,'+re.escape(dp)+r'})\d*$',r'\1',str(num)))

def plot_bar_x(measureType, key, value):
    # this is for plotting purpose
    index = np.arange(len(key))
    plt.figure()
    for i, v in enumerate(value):
        v = truncate(v,2)
        color = 'red'
        if v < 0.8:
            color = 'chocolate'
        if v >=0.8:
            color = 'olive'
        if v >= 0.89:
            color = 'olivedrab'
        if v > 0.9:
            color = 'limegreen'
        
        if i == 0:
            plt.bar(index[i], v, width = 0.9, edgecolor = 'black', ls = '-', lw = 1.5, color = color)
            plt.text(index[i] - 0.1, 0.3, str('%.2f'%v), fontsize = 12, rotation = 90)
        else:
            plt.bar(index[i], v, width = 0.9, edgecolor = 'black', ls = '--', lw = 0.5, color = color)
            plt.text(index[i] - 0.1, 0.3, str('%.2f'%v), fontsize = 12, rotation = 90)
#    plt.xlabel('Genre', fontsize=5)
    plt.ylabel('F-score', fontsize=20)
    plt.ylim([0.0,1.0])
    plt.xticks(index, key, fontsize=15, rotation=270)
    #plt.title('%s'%measureType)
    fig1 = plt.gcf()
    plt.draw()
    fig1.savefig(measureType, bbox_inches='tight')

if __name__ == '__main__':
    random.seed(42)

    if len(sys.argv) != 3:
        print("Please insert a file path to analyze as argument and an absolute path to save results")
        exit(1)


    input_file = pathlib.Path(str(sys.argv[1]))
    savepath = pathlib.Path(str(sys.argv[2]))
    input_name = input_file.stem
    nodename = input_file.stem.split('_')[0].split('/')[-1]

    data = pd.read_csv(input_file)
    #all the columns except the label one
    metricKeys = list(data.columns)[:-1]

    rus = RandomUnderSampler(sampling_strategy="majority", random_state=42)

    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

    resfile = savepath.joinpath(nodename + "_classification_results.csv")
    summarypng = savepath.joinpath(nodename + "_heatmap_summary.png")

    #select new data using only the most important columns
    features = data[list(data.columns)[:-1]]
    features = features.to_numpy()
    #select the labels
    labels = data[list(data.columns)[-1]]
    labels = labels.to_numpy()

    #60-40 train-test split
    numTrain = int(0.6*len(features))
    trainData = features[:numTrain]
    trainLbl = labels[:numTrain]
    testData = features[numTrain:]
    testLbl = labels[numTrain:]

    trainData, trainLbl = rus.fit_resample(trainData, trainLbl)
    testData, testLbl = rus.fit_resample(testData, testLbl)

    F = []

    #this time we perform training on 60% of the dataset, and classification on 40%
    clf.fit(trainData, trainLbl)
    pred = clf.predict(testData)
    print('- Classifier: %s' % clf.__class__.__name__)
    f1 = f1_score(testLbl, pred, average = 'weighted')
    print("Overall score: %f." % f1)
    F.append(f1)

    #calculate F1-score for each class
    test_all = f1_score(testLbl, pred, average=None)
    F.extend(list(test_all))

    for i, f in enumerate(test_all):
        print('Fault: %d,  F1: %f.'%(i,f))

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

    resDict = OrderedDict(zip(keys, map(lambda x: [x], F)))
    plotDict = dict()
    plotDict[nodename] = F
    res = pd.DataFrame(resDict, index=[nodename])

    res.to_csv(str(resfile))

    measureType = savepath.joinpath(nodename + "_barplot_summary.png")
    plot_bar_x(measureType, keys, F)
    plot_heatmap("F1-scores", plotDict, keys, summarypng, True)