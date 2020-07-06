"""
2020-07-06 10:43

@author: Vito Vincenzo Covella
"""

import numpy as np
import pandas as pd
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import os

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
    fileN_mostImportant = input_name + "_result_RFE_" + str(numImportantFeatures) + "mostImportant.txt"
    featureFile = input_name + "_RFE_mostImportantFeatures.txt"

    data = pd.read_csv(input_file)
    #all the columns except the label one
    metricKeys = list(data.columns)[:-1]

    #select the features (all the column except the label one)
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

    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

    #initialize RFE with RF classifier and numImportantFeatures to select
    rfe = RFE(clf, n_features_to_select=numImportantFeatures)

    X_rfe = rfe.fit_transform(trainData, trainLbl)

    #Fitting the data to model
    clf.fit(X_rfe, trainLbl)
    temp = pd.Series(rfe.support_,index = metricKeys)
    selected_features_rfe = temp[temp==True].index
    print("Selected features with RFE:")
    print(selected_features_rfe)

    with open(featureFile, 'w') as out:
        out.write('- Most important features:\n')

    #save most important columns in a text file
    with open(featureFile, 'a') as out:
        for col in selected_features_rfe:
            out.write('---- %s\n' % col)

    #select new data using only the most important columns
    features = data[selected_features_rfe.tolist()]
    features = features.to_numpy()
    #labels are the same as before (see above)
    trainData = features[:numTrain]
    #trainLbl are the same as before, no need to recompute them
    testData = features[numTrain:]
    #testLbl are the same as before, no need to recompute them

    F = []

    with open(fileN_mostImportant, 'w') as out:
        out.write('- Classifier: %s\n' % clf.__class__.__name__)

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
    measureType = input_name + "_result_RFE" + str(numImportantFeatures) + "mostImportant"
    plot_bar_x(measureType, keys, F)
