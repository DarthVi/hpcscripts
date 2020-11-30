import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import BayesianGaussianMixture

def mahalanobis(u, v, C):
    CI = np.linalg.inv(C)
    return distance.mahalanobis(u, v, CI)

def get_mahlanobis_distances(data, mixture_model):
    distances = [mahalanobis(x, mixture_model.means_[mixture_model.predict([x])[0]], mixture_model.covariances_[mixture_model.predict([x])[0]]) for x in data]
    return distances

def smooth_data(files, columns, faults):
    #my_bar = st.progress(0)
    length = len(files)
    cols_to_load = columns + ['faultLabel']

    meanlist = list()
    #perc_upgrade = int(100 / length)
    
    with st.spinner('Wait for computation...'):
        for file in files:
            #print(file)
            #read only the columns we are interested in
            n = pd.read_csv(file, usecols=cols_to_load, header=0, parse_dates=True)
            #select only the data relative to the faults we are interested in
            n = n[n['faultLabel'].isin(faults)]
            #print(n)
            #drop the fault label
            n.drop(['faultLabel'], axis=1, inplace=True)

            #reorder columns
            n = n[columns]

            #get the mean of every column, convert Series to dataframe and transpose
            #print(n.mean())
            n = n.mean().to_frame().T
            #print(n)

            #trick to insert nodename as first column
            n['nodename'] = file.stem
            nodename = n.pop('nodename')
            n.insert(0, 'nodename', nodename)
            meanlist.append(n)
            #my_bar.progress(perc_upgrade) 

    df = pd.concat(meanlist, axis=0).reset_index(drop=True)
    #print(df)
    #exit(1)
    return df

def mark_outliers(data, threshold):
    data.loc[df['distance'] > threshold, 'cluster'] = 'outlier'

def cluster_encode(data, outlier_str, label_col):
    encoder = LabelEncoder()
    encoder.fit(data[label_col])
    outlier_enc = encoder.transform([outlier_str])[0]
    new_enc = encoder.transform(data[label_col])
    data[label_col] = new_enc
    data.loc[data[label_col] == outlier_enc, label_col] = outlier_str
    data[label_col] = data[label_col].apply(lambda x: (x + 1) if not isinstance(x, str) else x)


def mark_outliers(data, threshold):
    df.loc[df['distance'] > threshold, 'cluster'] = 'outlier'


def plot_scatter(data, columns, cluster_str, outlier_str):
    colors = [x for x in px.colors.qualitative.Alphabet if x != '#F6222E'] #remove red, reserved for outliers
    fig = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2], color=cluster_str, symbol=cluster_str, color_discrete_sequence=colors)

    #if an entry is an outlier, plot it with a cross and use color red
    #otherwise use a circle and the corresponding cluster color
    for i, d in enumerate(fig.data):
        if fig.data[i].name == outlier_str:
            fig.data[i].marker.symbol = 'cross'
            fig.data[i].marker.color = '#F6222E'
        else:
            fig.data[i].marker.symbol = 'circle'

    st.plotly_chart(fig)

important_columns = ["thp_fault_alloc",
                     "instructions.max",
                     "opaif0/portXmitPackets",
                     "power",
                     "AnonPages",
                     "procs_running",
                     "ctxt",
                     "cache-references.max",
                     "instructions.perc75",
                     "branch-instructions.max",
                     "Active",
                     "pgfault"]

faults = ['healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

if __name__ == '__main__':
    st.title("Clustering HPC nodes")

    #select metrics we are interested in for clustering
    st.write("Select 3 different metrics to consider for the clustering")
    col1 = st.selectbox("Which metric on x?", important_columns)
    col2 = st.selectbox("Which metric on y?", important_columns)
    col3 = st.selectbox("Which metric on z?", important_columns)

    #select the which batch of data to consider (faulty or non faulty)
    st.write("Select the fault data to consider")
    fault_selected = st.multiselect("Which faults?", faults)
    fault_label_selected = [x for x, el in enumerate(faults) if el in fault_selected]

     #get folder in which CSV data of HPC nodes are stored
    folderPath = st.text_input('Enter folder path:')
    csvPath = pathlib.Path(folderPath)

    #get CSV filepath list
    nodefiles = list(csvPath.glob("*.csv"))
    num_files = len(nodefiles)

    n_components = st.slider("Select the number of components for the BGMM algorithm", min_value=1, max_value=num_files, value=10, step=1)
    threshold = st.number_input("Threshold to use for the outlier detection via Mahalanobis distance", step=0.00000001, format="%f", value=2.0)

    if st.button("Run"):
        nodefiles = sorted(nodefiles, key=lambda x: int(x.stem[1:]))

        smoothed_df = smooth_data(nodefiles, [col1, col2, col3], fault_label_selected)

        X = smoothed_df.drop(['nodename'], axis=1).to_numpy()

        bgmm = BayesianGaussianMixture(n_components=n_components, random_state=42)

        labels = bgmm.fit(X).predict(X)
        distances = get_mahlanobis_distances(X, bgmm)
        #create new dataframe with distances and clusters' labels
        df = pd.DataFrame({'nodename':smoothed_df['nodename'], col1: X[:, 0], col2: X[:, 1], col3: X[:, 2], 'distance': distances, 'cluster': labels})
        #transform the clusters' labels in string
        df['cluster'] = df['cluster'].astype(str)
        #mark the outliers in the data
        mark_outliers(df, threshold)

        #encode again the labels, in order to have consecutive ones like 1, 2, 3 ecc instead of 1, 5, 7, and so on
        cluster_encode(df, 'outlier', 'cluster')
        #transform again the clusters' labels in string since LabelEncoder gives integers
        df['cluster'] = df['cluster'].astype(str)

        #it will be used to order the DataFrame based on cluster labels
        clusters = set(np.unique(df['cluster'])) - {'outlier'}
        order_dict = dict()
        for c in clusters:
            order_dict[c] = int(c)
        order_dict['outlier'] = 0

        #sort the dataframe in order to plot the legend in an ordered way using plotly
        df.sort_values(by=['cluster'], key=lambda x: x.map(order_dict) , inplace=True)

        plot_scatter(df, [col1, col2, col3], 'cluster', 'outlier')

        st.write(df)