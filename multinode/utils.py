from collections import Counter
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

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

def truncate(num,decimal_places):
    dp = str(decimal_places)
    return float(re.sub(r'^(\d+\.\d{,'+re.escape(dp)+r'})\d*$',r'\1',str(num)))

def majority_mean(y):
    counter = Counter(y)
    dict_count = dict(counter)
    most_common_label = counter.most_common(1)[0][0]
    #get the rest of the counters (all the counters except the one of the most common label)
    rest_of_items = [x[1] for x in counter.items() if x[0] != most_common_label]
    mean = int(np.mean(rest_of_items))
    dict_count[most_common_label] = mean
    return dict_count

def minority_mean(y):
    counter = Counter(y)
    dict_count = dict(counter)
    least_common_label = counter.most_common()[-1][0]
    #get the rest of the counters (all the counters except the one of the most common label)
    rest_of_items = [x[1] for x in counter.items() if x[0] != least_common_label]
    mean = int(np.mean(rest_of_items))
    dict_count[least_common_label] = mean
    return dict_count


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_heatmap(title, vDict, columns, savepath, annotation=True):
    df = pd.DataFrame(vDict.values(), columns=columns, index=vDict.keys())
    df = df.applymap(lambda x: truncate(x, 2))
    if len(vDict) <= 6:
        figsize = (9, 6)
    else:
        figsize = (9, len(vDict)//1.5)
    # Draw a heatmap with the numeric values in each cell
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    sns.heatmap(df, annot=annotation, fmt=".2f", linewidths=.5, ax=ax, cmap=sns.cm.rocket_r, square=True, vmin=0, vmax=1)
    plt.yticks(rotation=0)
    fig.savefig(savepath, bbox_inches="tight")

def saveresults(rDict, columns, savepath):
    '''saves the classification results to CSV'''
    df = pd.DataFrame(rDict.values(), columns=columns, index=rDict.keys())
    df.to_csv(savepath, header=True, index=True)

#for each fault, update a CSV containing the fault scores for each node and the number of nodes used as training
def updateboxplotsCSV(dfpath, dfcolumn, savepath, numtrain, numrun=None):
    '''saves a CSV in the format that will be useful to later plot some boxplots about the F1-scores'''
    savefile = savepath.joinpath(dfcolumn + ".csv")
    includeHeader = not savefile.is_file()
    orig_df = pd.read_csv(dfpath, header=0, index_col=0)
    columnselected = orig_df[dfcolumn].reset_index(drop=True)
    label = pd.DataFrame({'num_train': [numtrain] * len(columnselected)})
    res = pd.concat([columnselected, label], axis=1)
    if numrun != None:
        numrun_label = pd.DataFrame({'num_iter': [numrun] * len(columnselected)})
        res = pd.concat([res, numrun_label], axis=1)
    res.to_csv(savefile, mode='a', index=False, header=includeHeader)

def summaryboxplot(readpath, column, savepath, label, title):
    '''Plot some summary boxplots for each column'''
    dfpath = readpath.joinpath(column + ".csv")
    saveimg = savepath.joinpath(column + ".png")
    df = pd.read_csv(dfpath, header=0)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=label, y=column, data=df)
    ax.set(ylim=(0.0, 1.0))
    ax.set_title(title)
    plt.draw()
    plt.savefig(saveimg, bbox_inches="tight")