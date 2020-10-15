"""
2020-10-02 14:06
@author: Vito Vincenzo Covella
"""

import numpy as np
import pandas as pd
import pathlib
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from FileFeatureReader.featurereaders import RFEFeatureReader
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
    #plt.show()
    plt.draw()
    fig1.savefig('%s'%(measureType), bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="backup_all_CSV/fvectors", help="Path in which there are the files to analyze")
    args = parser.parse_args()

    random.seed(42)

    here = pathlib.Path(__file__).parent #path of this script
    csvdir = here.joinpath(args.path) #get the directory in which there are the files to analyze
    summaryfile = csvdir.joinpath("results/summary.txt")
    summarypng = csvdir.joinpath("results/summary.png")

    rfe_file = csvdir.joinpath("RFEFeat.txt") #path of the file containing the features to extract
    rfe_feature_reader = FeatureReader(RFEFeatureReader(), str(rfe_file))
    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    experiments_scores = {}

    for file_entry in tqdm(list(csvdir.iterdir())):
        if file_entry.is_file() and file_entry.suffix == '.csv':
            filepath = csvdir.joinpath(file_entry.name)
            print("Processing file " + file_entry.name)
            input_name = file_entry.name.split('.')[0]
            nodename = file_entry.name.split('_')[0].split('/')[-1]
            five_fold_out = csvdir.joinpath("results/" + input_name + "_result_5fold.txt")
            img_filename = csvdir.joinpath("results/" + input_name + "_result_image.png")
            
            #get the dataframe considering only specific columns
            data = pd.read_csv(file_entry, usecols=selectionlist)
            X = data[featurelist].to_numpy()
            y = data['label'].to_numpy()
            #labels = np.unique(y)
            #random undersampler resample only the majority class
            rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)

            #classifier model
            clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

            kf = KFold(n_splits=5)
            #array to memorize the scores
            scoreArray = np.zeros(shape=(5,9), dtype=np.float64, order='C')
            for fold, (train_index, test_index) in enumerate(kf.split(X), 0):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_train, y_train = rus.fit_resample(X_train, y_train)
                X_test, y_test = rus.fit_resample(X_test, y_test)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                test_weighted = f1_score(y_test, y_pred, average="weighted")
                test_all = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=1)
                scoreArray[fold, 0] = test_weighted
                scoreArray[fold, 1:] = test_all

            keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']
            overall_score = scoreArray[:,0].mean()
            
            key_score = [scoreArray[:,i].mean() for i in range(1, 9)]
            all_scores = [overall_score] + key_score

            with open(five_fold_out, 'w') as out:
                out.write('- Classifier: %s\n' % clf.__class__.__name__)

            for f, v in enumerate(key_score, 0):
                print('Fault: %s,  F1: %s' % (f, v))
                with open(five_fold_out, 'a') as out:
                    out.write('Fault: %s,  F1: %s\n' % (f, v))

            print("---------------")

            plot_bar_x(str(img_filename), keys, all_scores)

            #save overall score in dictionary with nodename as key
            experiments_scores[nodename] = overall_score

    sort_expscores = sorted(experiments_scores.items(), key=lambda x: x[1], reverse=True)
    with open(summaryfile, 'w') as out:
        out.write("Overall scores for %s\n---------------\n" % clf.__class__.__name__)
        for i in sort_expscores:
            out.write("Node: %s,  F1:%s\n" % (i[0], i[1]))

    #summary bar plot
    plt.figure()
    plt.bar(*zip(*experiments_scores.items()))
    plt.draw()
    plt.savefig('%s' % str(summarypng))
