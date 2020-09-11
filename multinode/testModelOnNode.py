"""
2020-09-10 16:39

@author: Vito Vincenzo Covella
"""

import numpy as np
import pandas as pd
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import os
from joblib import load
from FileFeatureReader.featurereaders import RFEFeatureReader, DTFeatureReader
from FileFeatureReader.featurereader import FeatureReader

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

    if len(sys.argv) != 4:
        print("Please insert a model file to load, a feature list file and a node file on which to test the model")
        exit(1)

    #model to load
    input_file = str(sys.argv[1])
    input_name = input_file.split('.')[0]
    nodename = input_file.split('_')[0].split('/')[-1]

    #text file containing features to consider
    featurepath = str(sys.argv[2])
    rfe_feature_reader = FeatureReader(RFEFeatureReader(), featurepath)

    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    #test node file
    testnode = str(sys.argv[3])
    testnode_name = testnode.split('.')[0]
    testnode_nodename = input_file.split('_')[0].split('/')[-1]

    resultOut = testnode_name + "_result.txt"

    clf = load(input_file)

    #get the dataframe considering only specific columns
    data = pd.read_csv(testnode, usecols=selectionlist)

    features = data[featurelist]
    labels = data['label']

    #list to store f1 values
    F = []

    with open(resultOut, 'w') as out:
       out.write('- Classifier: %s\n' % clf.__class__.__name__)

    pred = clf.predict(features)
    print('- Classifier: %s' % clf.__class__.__name__)
    f1 = f1_score(labels, pred, average = 'weighted')
    F.append(f1)

    for l in np.unique(np.asarray(labels)):
        #select the correct labels
        lab_tmp = labels[list(np.where(labels == l)[0])]
        #select the corresponding predicted lables
        pred_tmp = pred[list(np.where(labels == l)[0])]
        f1 = f1_score(lab_tmp, pred_tmp, average = 'micro')
        F.append(f1)
        print('Fault: %d,  F1: %f.\n'%(l,f1))
        with open(resultOut, 'a') as out:
            out.write('Fault: %d,  F1: %f.\n'%(l,f1))

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']
    measureType = testnode_name + "_result_image"
    plot_bar_x(measureType, keys, F)