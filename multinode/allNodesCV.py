"""
2020-10-16 12:52
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
from FileFeatureReader.featurereaders import RFEFeatureReader
from FileFeatureReader.featurereader import FeatureReader

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
    if len(vDict) <= 6:
        figsize = (9, 6)
    else:
        figsize = (9, len(vDict)//1.5)
    # Draw a heatmap with the numeric values in each cell
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    sns.heatmap(df, annot=annotation, fmt=".2g", linewidths=.5, ax=ax)
    plt.yticks(rotation=0)
    fig.savefig(savepath, bbox_inches="tight")

if __name__ == '__main__':
    random.seed(42)