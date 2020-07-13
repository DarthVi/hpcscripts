"""
2020-07-03 16:28
@author: Vito Vincenzo Covella
"""

import numpy as np
import pandas as pd
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
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

def getScorerObjects(labels):
    """
    Creates a dictionary of SciKit scorer objects. Each object considers features from a specific class out of those
    given as input, and the metric used here is the F-Score

    :param labels: The list of class labels
    :return: A list of Scikit scorer objects
    """
    labelSet = set(labels)
    scorers = {}
    for label in labelSet:
        scorers[str(label)] = make_scorer(f1_score, average=None, labels=[label])
    scorers['weighted'] = make_scorer(f1_score, average='weighted')
    return scorers

def main():
    random.seed(42)

    if len(sys.argv) != 2:
        print("Please insert a file path to analyze as argument")
        exit(1)

    input_file = str(sys.argv[1])
    input_name = input_file.split('.')[0]
    nodename = input_file.split('_')[0].split('/')[-1]
    five_fold_out = input_name + "_result_5fold.txt"

    data = pd.read_csv(input_file)
    #all the columns except the label one
    metricKeys = list(data.columns)[:-1]

    #select the features (all the column except the label one)
    features = data[list(data.columns)[:-1]]
    features = features.to_numpy()
    #select the labels
    labels = data[list(data.columns)[-1]]
    labels = labels.to_numpy()

    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=None, random_state=42)

    scorers = getScorerObjects(labels)
    scores = cross_validate(clf, features, labels, cv=5, scoring=scorers, n_jobs=-1)

    #print(scores)

    F = []

    print('- Classifier: %s' % clf.__class__.__name__)
    #take the mean of the 5fold
    mean = scores['test_weighted'].mean()
    #confidence = scores['test_weighted'].std() * 1.96 / np.sqrt(len(scores['test_weighted']))
    #print('- Global F-Score : %s (+/- %s)' % (mean, confidence))
    print('- Global F-Score : %s' % mean)

    with open(five_fold_out, 'w') as out:
        out.write('- Classifier: %s\n' % clf.__class__.__name__)

    F.append(mean)

    for k, v in scores.items():
       if 'test_' in k and k != 'test_weighted':
        #take the mean of the 5fold
           mean = scores[k].mean()
           F.append(mean)
           #confidence = scores[k].std() * 1.96 / np.sqrt(len(scores[k]))
           #print('---- %s F-Score : %s (+/- %s)' % (k.split('_')[1], mean, confidence))
           print('Fault: %s,  F1: %s' % (int(float(k.split('_')[1])), mean))
           with open(five_fold_out, 'a') as out:
            out.write('Fault: %s,  F1: %s\n' % (int(float(k.split('_')[1])), mean))

    print('---------------')

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']
    measureType = input_name + "_result_5fold"
    plot_bar_x(measureType, keys, F)

if __name__ == '__main__':
    main()

