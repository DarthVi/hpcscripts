from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

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
