import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

def mahalanobis(u, v, C):
    CI = np.linalg.inv(C)
    return distance.mahalanobis(u, v, CI)

@st.cache(suppress_st_warning=True)
def saveontextfile(strlst, path):
    with open(path, 'w') as out:
        for s in strlst:
            out.write(s + '\n')

@st.cache(suppress_st_warning=True)
def saveoutlierthreshold(value, path):
    with open(path, 'w') as out:
        if value != np.inf:
            out.write("{:.8f}\n".format(value))
        else:
            out.write(str(np.inf)+"\n")

@st.cache(suppress_st_warning=True)
def savefigure(fig, path):
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    fig.write_image(path)

@st.cache(suppress_st_warning=True)
def savedataframe(df, path):
    df.to_csv(path, header=True, index=True)

@st.cache(suppress_st_warning=True) 
def get_mahlanobis_distances(data, mixture_model, labs):
    #distances = [mahalanobis(x, mixture_model.means_[mixture_model.predict([x])[0]], mixture_model.covariances_[mixture_model.predict([x])[0]]) for x in data]
    distances = [mahalanobis(x, mixture_model.means_[l], mixture_model.covariances_[l]) for x,l in zip(data, labs)]
    return distances

@st.cache(suppress_st_warning=True)
def smooth_data_PCA(files, pcamodel, faults, apps):
    my_bar = st.progress(0)
    metriclist = list()

    meanlist = list()
    files_len = len(files)
    
    with st.spinner('Wait for computation...'):
        for i,file in enumerate(files):
            #read only the columns we are interested in
            n = pd.read_csv(file, index_col=0, parse_dates=True, header=0)
            #initialize metric list if this is the first file processed
            if not metriclist:
                metriclist = list(n.columns)
                metriclist.remove('faultLabel')
                metriclist.remove('applicationLabel')

            #save faultLabels and applicationLabel infos
            labels = n[['faultLabel', 'applicationLabel']].reset_index(drop=True)
            #drop the fault label
            n.drop(['faultLabel', 'applicationLabel'], axis=1, inplace=True)
            #reorder columns and get numpy values
            n = n[metriclist].to_numpy()
            n = StandardScaler().fit_transform(n)
            #perform PCA
            n = pcamodel.fit_transform(n)
            n = pd.DataFrame(data = n, columns = ['pc1', 'pc2', 'pc3'])
            #attach again label infos
            n = pd.concat([n, labels], axis=1)
            #select only the data relative to the faults we are interested in
            n = n[n['faultLabel'].isin(faults)]
            #select only the data relative to the applications we are interested in
            n = n[n['applicationLabel'].isin(apps)]

            #drop again labels
            n.drop(['faultLabel', 'applicationLabel'], axis=1, inplace=True)

            #get the mean of every column, convert Series to dataframe and transpose
            n = n.mean().to_frame().T

            #trick to insert nodename as first column
            n['nodename'] = file.stem
            nodename = n.pop('nodename')
            n.insert(0, 'nodename', nodename)
            meanlist.append(n)
            my_bar.progress((i+1)/files_len) 

    df = pd.concat(meanlist, axis=0).reset_index(drop=True)
    my_bar.empty()
    return df

@st.cache(suppress_st_warning=True) 
def smooth_data(files, columns, faults, apps):
    my_bar = st.progress(0)
    cols_to_load = columns + ['faultLabel', 'applicationLabel']

    meanlist = list()
    files_len = len(files)
    
    with st.spinner('Wait for computation...'):
        for i,file in enumerate(files):
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
            meanlist.append(n)
            my_bar.progress((i+1)/files_len) 

    df = pd.concat(meanlist, axis=0).reset_index(drop=True)
    my_bar.empty()
    return df


def mark_outliers(data, threshold):
    data.loc[df['distance'] > threshold, 'cluster'] = 'outlier'
    return data


def cluster_encode(data, outlier_str, label_col):
    unique_labels = np.unique(data[label_col])
    encoder = LabelEncoder()
    encoder.fit(data[label_col])
    if outlier_str in unique_labels:
        outlier_enc = encoder.transform([outlier_str])[0]
    new_enc = encoder.transform(data[label_col])
    data[label_col] = new_enc
    if outlier_str in unique_labels:
        data.loc[data[label_col] == outlier_enc, label_col] = outlier_str
    data[label_col] = data[label_col].apply(lambda x: (x + 1) if not isinstance(x, str) else x)


def plot_scatter(data, columns, cluster_str, outlier_str):
    colors = [x for x in px.colors.qualitative.Alphabet if x != '#F6222E'] #remove red, reserved for outliers
    fig = px.scatter_3d(data, x=columns[0], y=columns[1], z=columns[2], color=cluster_str, symbol=cluster_str, color_discrete_sequence=colors, 
        hover_name="nodename", hover_data=['distance'])

    #if an entry is an outlier, plot it with a cross and use color red
    #otherwise use a circle and the corresponding cluster color
    for i, d in enumerate(fig.data):
        if fig.data[i].name == outlier_str:
            fig.data[i].marker.symbol = 'cross'
            fig.data[i].marker.color = '#F6222E'
        else:
            fig.data[i].marker.symbol = 'circle'

    #fig.update_layout(margin=dict(l=5,r=5,t=5,b=5))
    st.plotly_chart(fig)
    return fig

@st.cache(suppress_st_warning=True) 
def compute_GMM(data, n_components, column1, column2, column3):
    X = data.drop(['nodename'], axis=1).to_numpy()

    bgmm = GaussianMixture(n_components=n_components, random_state=42)

    labels = bgmm.fit(X).predict(X)
    distances = get_mahlanobis_distances(X, bgmm, labels)
    #create new dataframe with distances and clusters' labels
    df = pd.DataFrame({'nodename':data['nodename'], column1: X[:, 0], column2: X[:, 1], column3: X[:, 2], 'distance': distances, 'cluster': labels})
    #transform the clusters' labels in string
    df['cluster'] = df['cluster'].astype(str)
    return df


def stringify_cluster(data):
    data['cluster'] = data['cluster'].astype(str)
    return data


def order_dataframe_by_cluster(data):
    #it will be used to order the DataFrame based on cluster labels
    clusters = set(np.unique(data['cluster'])) - {'outlier'}
    order_dict = dict()
    for c in clusters:
        order_dict[c] = int(c)
    order_dict['outlier'] = 0

    #sort the dataframe in order to plot the legend in an ordered way using plotly
    data.sort_values(by=['cluster'], key=lambda x: x.map(order_dict) , inplace=True)
    return data

def get_prototypes(data):
    data = data[data['cluster'] != 'outlier']
    data = data.groupby('cluster').min()
    return data

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

applications = {
    'idle': 0,
    'Kripke': 20,
    'AMG': 21,
    'Nekbone': 22,
    'PENNANT': 23,
    'HPL': 24
}

if __name__ == '__main__':
    st.title("Clustering HPC nodes")

    method = st.radio("How to select columns:", ("PCA", "manually"), index=1)
    if method == 'manually':
        #select metrics we are interested in for clustering
        st.write("Select 3 different metrics to consider for the clustering")
        col1 = st.selectbox("Which metric on x?", important_columns)
        col2 = st.selectbox("Which metric on y?", important_columns)
        col3 = st.selectbox("Which metric on z?", important_columns)
    else:
        col1 = 'pc1'
        col2 = 'pc2'
        col3 = 'pc3'

    #select the which batch of data to consider (faulty or non faulty)
    st.write("Select the fault data to consider")
    fault_selected = st.multiselect("Which faults?", faults, default=faults[0])
    fault_label_selected = [x for x, el in enumerate(faults) if el in fault_selected]

    st.write("Select the applications you are interested in")
    app_keys = list(applications.keys())
    app_selected = st.multiselect("Which applications?", app_keys, default=app_keys)
    app_label_selected = [applications[x] for x in app_selected]

    #get folder in which CSV data of HPC nodes are stored
    folderPath = st.text_input('Enter folder path:')
    csvPath = pathlib.Path(folderPath)

    here = pathlib.Path(__file__).parent
    defaultSavepath = here.joinpath("results")
    #get the folder path in which to save the results
    saveP = st.text_input('Enter the folder in which to save the results:', value=str(defaultSavepath))
    savepath = pathlib.Path(saveP)

    #get CSV filepath list
    nodefiles = list(csvPath.glob("*.csv"))
    num_files = len(nodefiles)

    n_components = st.slider("Select the number of components for the GMM algorithm", min_value=1, max_value=num_files, value=10, step=1)
    detect_outliers = st.radio("Detect outliers?", ('yes', 'no'), index=1)
    if detect_outliers == 'yes':
        threshold = st.number_input("Threshold to use for the outlier detection via Mahalanobis distance", step=0.00000001, format="%f", value=2.0)
    else:
        threshold = np.inf

    if st.button("Run"):
        nodefiles = sorted(nodefiles, key=lambda x: int(x.stem[1:]))

        if method == 'manually':
            smoothed_df = smooth_data(nodefiles, [col1, col2, col3], fault_label_selected, app_label_selected)
            #r_covar = 1e-6
        else:
            pca = PCA(n_components=3)
            smoothed_df = smooth_data_PCA(nodefiles, pca, fault_label_selected, app_label_selected)
            #r_covar = 1e-2

        df = compute_GMM(smoothed_df, n_components, col1, col2, col3)

        #mark the outliers in the data
        df_marked = mark_outliers(deepcopy(df), threshold)

        #encode again the labels, in order to have consecutive ones like 1, 2, 3 ecc instead of 1, 5, 7, and so on
        cluster_encode(df_marked, 'outlier', 'cluster')
        #transform again the clusters' labels in string since LabelEncoder gives integers
        df_marked = stringify_cluster(df_marked)

        df_marked = order_dataframe_by_cluster(df_marked)

        figure = plot_scatter(df_marked, [col1, col2, col3], 'cluster', 'outlier')

        st.subheader("Resulting dataframe after clustering")
        st.write(df_marked)

        prototype_df = get_prototypes(deepcopy(df_marked))

        st.subheader("Prototypes for each cluster")
        st.write(prototype_df)

        savepath.mkdir(parents=True, exist_ok=True)
        saveontextfile(fault_selected, savepath.joinpath("fault_selected.txt"))
        saveontextfile(app_selected, savepath.joinpath("app_selected.txt"))
        saveontextfile([col1, col2, col3], savepath.joinpath("col_selected.txt"))
        saveoutlierthreshold(threshold, savepath.joinpath("outlier_threshold.txt"))
        savefigure(deepcopy(figure), str(savepath.joinpath("scatter_plot.png")))
        savedataframe(df_marked, savepath.joinpath("clustered_df.csv"))
        savedataframe(prototype_df, savepath.joinpath("candidate_nodes.csv"))
