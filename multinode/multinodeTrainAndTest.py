"""
2020-10-03 15:04
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
import seaborn as sns
import os
from collections import OrderedDict
import argparse
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
from joblib import dump
from FileFeatureReader.featurereaders import RFEFeatureReader
from FileFeatureReader.featurereader import FeatureReader

def plot_heatmap(title, vDict, columns, savepath):
    df = pd.DataFrame(vDict.values(), columns=columns, index=vDict.keys())
    # Draw a heatmap with the numeric values in each cell
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(title)
    sns.heatmap(df, annot=True, fmt=".2g", linewidths=.5, ax=ax)
    plt.yticks(rotation=0)
    fig.savefig(savepath, bbox_inches="tight")

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

def saveresults(rDict, columns, savepath):
    '''saves the classification results to CSV'''
    df = pd.DataFrame(rDict.values(), columns=columns, index=rDict.keys())
    df.to_csv(savepath, header=True, index=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--trainpath", type=str, default="train/", help="Path in which there are the training files")
    parser.add_argument("-t", "--testpath", type=str, default="test/", help="Path in which there are the test files")
    parser.add_argument("-r", "--savepath", type=str, default="results/", help="Path in which to store the results")
    parser.add_argument("-s", "--sampling", type=str, default="majority", help="Undersampling strategy for the random undersampler (for class balancing)")
    args = parser.parse_args()

    random.seed(42)

    here = pathlib.Path(__file__).parent #path of this script
    trainpath = here.joinpath(args.trainpath) #path in which there are the training files
    testpath = here.joinpath(args.testpath) #path in which there are the testing files
    resultspath = here.joinpath(args.savepath) #path of the results folder
    rfe_file = trainpath.joinpath("RFEFeat.txt")

    rfe_feature_reader = FeatureReader(RFEFeatureReader(), rfe_file)

    #get features used for training
    featurelist = rfe_feature_reader.getFeats()
    #add the label to the previous list
    selectionlist = featurelist + ['label']

    train_nodenames = list()
    dfs = list()

    #loop over all the CSV containing training data, in order to extract it
    for file_entry in tqdm(list(trainpath.iterdir())):
        if file_entry.is_file() and file_entry.suffix == '.csv':
            filepath = trainpath.joinpath(file_entry.name)
            print("Getting data from file ", file_entry.name)
            input_name = file_entry.name.split('.')[0]
            nodename = file_entry.name.split('_')[0].split('/')[-1]
            train_nodenames.append(nodename)

            #get the dataframe considering only specific columns
            data = pd.read_csv(file_entry, usecols=selectionlist)
            dfs.append(data)

    #concatenate all the training dataframes in one dataframe
    train_df = pd.concat(dfs, ignore_index=True)
    X = train_df[featurelist].to_numpy()
    y = train_df['label'].to_numpy()

    #classifier model
    clf = RandomForestClassifier(n_estimators=30, max_depth=20, n_jobs=-1, random_state=42)

    #random undersampling for class balancing
    rus = RandomUnderSampler(sampling_strategy=args.sampling, random_state=42)

    #train the model
    X, y = rus.fit_resample(X, y)
    clf.fit(X, y)

    #save the model
    model_savefile = '_'.join(train_nodenames) + "_model.joblib"
    model_savepath = resultspath.joinpath(model_savefile)
    dump(clf, model_savepath)

    keys = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']
    clsResults = OrderedDict()

    for file_entry in tqdm(list(testpath.iterdir())):
        if file_entry.is_file() and file_entry.suffix == '.csv':
            filepath = testpath.joinpath(file_entry.name)
            print("Testing on file ", file_entry.name)
            input_name = file_entry.name.split('.')[0]
            nodename = file_entry.name.split('_')[0].split('/')[-1]

            #get the dataframe considering only specific columns
            data = pd.read_csv(file_entry, usecols=selectionlist)
            X = data[featurelist].to_numpy()
            y = data['label']
            labels = np.unique(np.asarray(y))
            y = y.to_numpy()

            #class balancing by random undersampling
            X, y = rus.fit_resample(X, y)

            pred = clf.predict(X)

            #list to store f1 values
            F = []

            print('- Classifier: %s' % clf.__class__.__name__)

            #calculate global overall F1-score with weighted average
            f1 = f1_score(y, pred, average = 'weighted')
            F.append(f1)
            print('Overall score: %f.'%f1)

            #calculate score for each class by micro-averaging
            for l in np.unique(np.asarray(labels)):
                #select the correct labels
                lab_tmp = y[list(np.where(y == l)[0])]
                #select the corresponding predicted lables
                pred_tmp = pred[list(np.where(y == l)[0])]
                f1 = f1_score(lab_tmp, pred_tmp, average = 'micro')
                F.append(f1)
                print('Fault: %d,  F1: %f.'%(l,f1))

            clsResults[nodename] = F.copy()

    #construct path for storing the result summary in txt
    resultOut = "classification_results.csv"
    result_path = resultspath.joinpath(resultOut)
    #construct path for saving png about classification performances
    measureType = "classification_results_image.png"
    result_image = resultspath.joinpath(measureType)

    saveresults(clsResults, keys, result_path)
    plot_heatmap(clsResults, keys, result_image)