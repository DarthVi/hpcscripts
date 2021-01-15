import pathlib
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pprint import pprint

from tqdm import tqdm

def smooth_data(files, columns, faults, apps):
    cols_to_load = columns + ['faultLabel', 'applicationLabel']

    meanlist = list()
    files_len = len(files)
    
    for i,file in tqdm(enumerate(files), total=files_len):
        #read only the columns we are interested in
        n = pd.read_csv(file, usecols=cols_to_load, header=0)
        #select only the data relative to the faults we are interested in
        n = n[n['faultLabel'].isin(faults)]
        #select only the data relative to the applications we are interested in
        n = n[n['applicationLabel'].isin(apps)]
        
        #drop the fault label
        n.drop(['faultLabel', 'applicationLabel'], axis=1, inplace=True)

        #reorder columns
        n = n[columns]

        #get the mean of every column, convert Series to dataframe and transpose
        n = n.mean().to_frame().T

        #trick to insert nodename as first column
        n['nodename'] = file.stem
        nodename = n.pop('nodename')
        n.insert(0, 'nodename', nodename)
        n['method'] = 'random'
        meanlist.append(n)

    df = pd.concat(meanlist, axis=0).reset_index(drop=True)
    return df

def savefigure(fig, path):
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    fig.write_image(path)

def select_random_training_nodes(rdf, allnodes, n_train, t_nodes):
    selected_nodes = list()
    test_nodes = t_nodes - n_train
    start = 0
    stop = test_nodes

    while stop <= test_nodes * 10:
        testing = list(rdf.iloc[start:stop].index)
        rand_n = set(allnodes) - set(testing)
        selected_nodes.extend(rand_n)
        start = stop
        stop = stop + test_nodes

    return list(set(selected_nodes))



if __name__ == '__main__':
    dfpath = pathlib.Path(str(sys.argv[1]))
    randompath = pathlib.Path(str(sys.argv[2]))
    csvpath = pathlib.Path(str(sys.argv[3]))
    spath = pathlib.Path(str(sys.argv[4]))
    num_train = int(sys.argv[5])
    total_nodes = int(sys.argv[6])
    savefile = spath.joinpath("clustering_vs_random_scatterplot.png")

    cols = ['cache-misses.min', 'instructions.max', 'power']
    cols_plus_nodename = ['nodename'] + cols

    clust_df = pd.read_csv(dfpath, header=0, index_col=0)
    clust_df = clust_df[cols_plus_nodename]
    clust_df['method'] = 'clustering'
    rand_res = pd.read_csv(randompath, header=0, index_col=0)

    faults = ['healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

    applications = {
        'idle': 0,
        'Kripke': 20,
        'AMG': 21,
        'Nekbone': 22,
        'PENNANT': 23,
        'HPL': 24
    }

    fault_selected = ["healthy"]
    fault_label_selected = [x for x, el in enumerate(faults) if el in fault_selected]
    app_selected = ["HPL"]
    app_label_selected = [applications[x] for x in app_selected]

    nodefiles = list(csvpath.glob("*.csv"))
    nodefiles = sorted(nodefiles, key=lambda x: int(x.stem[1:]))
    nodenames = list(map(lambda x: x.stem, nodefiles))

    random_nodes = select_random_training_nodes(rand_res, nodenames, num_train, total_nodes)

    pprint(random_nodes)

    random_nodes_files = list(map(lambda x: csvpath.joinpath(x + ".csv"), random_nodes))

    random_nodes_data = smooth_data(random_nodes_files, cols, fault_label_selected, app_label_selected)

    fdf = pd.concat([clust_df, random_nodes_data], axis=0)

    #idx = fdf['nodename'].duplicated().index
    #fdf[fdf['nodename'].duplicated(), 'method'] = 'both'
    fdf.loc[fdf['nodename'].duplicated(), 'method'] = 'both'
    print(fdf)
    #print(idx)
    #fdf.iloc[idx, 'method'] = "both"

    colors = [x for x in px.colors.qualitative.Alphabet]
    figure = px.scatter_3d(fdf, x=cols[0], y=cols[1], z=cols[2], color='method', color_discrete_sequence=colors, 
        hover_name="nodename")

    savefigure(figure, str(savefile))