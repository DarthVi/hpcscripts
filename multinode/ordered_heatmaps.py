from collections import Counter
import re
import sys
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def truncate(num,decimal_places):
    dp = str(decimal_places)
    return float(re.sub(r'^(\d+\.\d{,'+re.escape(dp)+r'})\d*$',r'\1',str(num)))

def plot_heatmap_light(title, df, savepath, annotation=True):
    df = df.applymap(lambda x: truncate(x, 2))
    if len(df) <= 6:
        figsize = (9, 6)
    else:
        figsize = (9, len(df)//1.5)
    # Draw a heatmap with the numeric values in each cell
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    sns.heatmap(df, annot=annotation, fmt=".2f", linewidths=.5, ax=ax, cmap=sns.cm.rocket, square=True, vmin=0, vmax=1)
    plt.yticks(rotation=0)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()

def plot_heatmap(title, df, savepath, annotation=True):
    df = df.applymap(lambda x: truncate(x, 2))
    if len(df) <= 6:
        figsize = (9, 6)
    else:
        figsize = (9, len(df)//1.5)
    # Draw a heatmap with the numeric values in each cell
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    sns.heatmap(df, annot=annotation, fmt=".2f", linewidths=.5, ax=ax, cmap=sns.cm.rocket_r, square=True, vmin=0, vmax=1)
    plt.yticks(rotation=0)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    
    dfpath = pathlib.Path(str(sys.argv[1]))
    parentfolder = dfpath.parent
    newcsv = parentfolder.joinpath("classification_results_ordered.csv")
    savefile = parentfolder.joinpath("summary.png")
    savefile_light = parentfolder.joinpath("summary_light.png")
    title = str(sys.argv[2])

    df = pd.read_csv(dfpath, header=0, index_col=0)

    odf = df.sort_index(key=lambda x: pd.Index([int(y[1:]) for y in x]))

    odf.to_csv(newcsv, header=True, index=True)

    plot_heatmap(title, odf, savefile)
    plot_heatmap_light(title, odf, savefile_light)