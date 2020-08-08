"""
2020-07-04 15:25

@author: Vito Vincenzo Covella
"""

import numpy as np
import dask.dataframe as dd
import dask.array as da
import pandas as pd
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import os

#######functions to count lines of a file, from https://stackoverflow.com/a/27518377/1262118
def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum( buf.count(b'\n') for buf in f_gen )
########

def plot_bar_x(measureType, key, value):
    # this is for plotting purpose
    index = np.arange(len(key))
    plt.figure()
    for i, v in enumerate(value):
        v = np.round(v,2)
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
    plt.show()
    plt.draw()
    fig1.savefig('%s.png'%(measureType), bbox_inches='tight')

if __name__ == '__main__':

    random.seed(42)

    if len(sys.argv) != 3:
        print("Please insert a file path to analyze as argument and the number of important features to select")
        exit(1)

    numImportantFeatures = int(str(sys.argv[2]))

    input_file = str(sys.argv[1])
    input_name = input_file.split('.')[0]
    nodename = input_file.split('_')[0].split('/')[-1]
    fileN_mostImportant = input_name + "_result_" + str(numImportantFeatures) + "mostImportant.txt"
    featureFile = input_name + "_mostImportantFeatures.txt"

    len_features = rawgencount(input_file)

    #print(nodename)

    data = dd.read_csv(input_file, header=0)
    #all the columns except the label one
    metricKeys = list(data.columns)[:-1]

    #select the features (all the column except the label one)
    features = data[list(data.columns)[:-1]]
    features = features.to_dask_array(lengths=True)
    #select the labels
    labels = data[list(data.columns)[-1]]
    labels = labels.to_dask_array(lengths=True)
    #60-40 train-test split
    numTrain = int(0.6*len_features)
    trainData = features[:numTrain]
    trainLbl = labels[:numTrain].compute()
    #testData = features[numTrain:]
    #testLbl = labels[numTrain:]

    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)


    with open(featureFile, 'w') as out:
        out.write('- Most important features:\n')

    clf.fit(trainData, trainLbl)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('- Most important features:')
    with open(featureFile, 'a') as out:
       for idx in range(len(indices)):
           metric = indices[idx]
           print('---- %s: %s' % (metricKeys[metric], importances[metric]))
           out.write('---- %s: %s\n' % (metricKeys[metric], importances[metric]))

    #retrain with most important features
    mostImportantColumns = []
    for i in range(numImportantFeatures):
        metric = indices[i]
        mostImportantColumns.append(metricKeys[metric])


    #select features vectors using only the k most important features
    features = data[mostImportantColumns]
    features = features.to_dask_array(lengths=True)
    trainData = features[:numTrain]
    trainLbl = labels[:numTrain].compute()
    testData = features[numTrain:]
    testLbl = labels[numTrain:].compute()

    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

    with open(fileN_mostImportant, 'w') as out:
        out.write('- Classifier: %s\n' % clf.__class__.__name__)

    F = []

    clf.fit(trainData, trainLbl)
    pred = clf.predict(testData)
    print('- Classifier: %s' % clf.__class__.__name__)
    f1 = f1_score(testLbl, pred, average = 'weighted')
    F.append(f1)

    for l in np.unique(np.asarray(testLbl)):
        #select the correct labels from testLbl
        lab_tmp = testLbl[list(np.where(testLbl == l)[0])]
        #select the corresponding predicted lables
        pred_tmp = pred[list(np.where(testLbl == l)[0])]
        f1 = f1_score(lab_tmp, pred_tmp, average = 'micro')
        F.append(f1)
        print('Fault: %d,  F1: %f.\n'%(l,f1))
        with open(fileN_mostImportant, 'a') as out:
            out.write('Fault: %d,  F1: %f.\n'%(l,f1))



    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']
    measureType = input_name + "_result_" + str(numImportantFeatures) + "mostImportant"
    plot_bar_x(measureType, keys, F)