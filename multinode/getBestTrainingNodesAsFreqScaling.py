import random

from tqdm import tqdm
import numpy as np
from numpy import random as rnp
import pandas as pd
import pathlib
from tslearn.utils import to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import argparse

#remove numpy array from list of numpy arrays
def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="relative path in which there are the files to analyze")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed used by the script")
    parser.add_argument("-k", "--clusters", type=int, default=4, help="How many representative nodes to take")
    parser.add_argument("-v", "--savepath", type=str, default="results/", help="relative path of the folder in which to save the experiment results")
    parser.add_argument("-t", "--freq", type=str, default="1T", help="The sampling frequency used to select data from the dataset")
    args = parser.parse_args()

    rnp.seed(args.seed)
    random.seed(args.seed)

    here = pathlib.Path(__file__).parent #path of this script
    csvpath = here.joinpath(args.path) #get the directory in which there are the files to analyze
    resultsfile = here.joinpath(args.savepath).joinpath("nodes_selected.csv")

    csvlist = csvpath.glob("*.csv")
    csvlist = sorted(csvlist, key=lambda x: int(x.stem.split('_')[0][1:]))

    #nodedict = dict()
    selectedcols = list()
    dataset = list()
    for file_entry in tqdm(list(csvlist)):
        if not selectedcols:
            selectedcols.extend(list(pd.read_csv(file_entry, header=0, index_col=0, parse_dates=True, nrows=1).columns.drop(['faultLabel'])))
        #skip first 5 hours, take next 2 hours of data
        df = pd.read_csv(file_entry, header=0, index_col=0, parse_dates=True, usecols=['Time'] + selectedcols)
        X = df.asfreq(args.freq)
        X = X[selectedcols]
        X = X.to_numpy()
        #trick to associate numpy arrays to node names: ndarray.tobytes() will return a raw python bytes string which is immutable
        #nodedict[X.tobytes()] = file_entry.stem
        dataset.append(X)
        del df
        del X

    print("Creating tslearn dataset")
    #get dataset as array with shape (n_ts, max_sz, d), as expected by tslearn library, see tslearn documentation
    dataset = to_time_series_dataset(dataset)
    backup_dataset = np.copy(dataset)
    dataset = TimeSeriesScalerMeanVariance().fit_transform(dataset)
    print("Dataset shape", dataset.shape)


    print("Performing clustering and finding centroids")
    #performs KMeans clustering with DTW as distance measure
    km_dba = TimeSeriesKMeans(n_clusters=args.clusters, metric="dtw", n_jobs=-1, random_state=args.seed, max_iter=10, max_iter_barycenter=30).fit(dataset)
    #get the centroids
    centroids = km_dba.cluster_centers_
    print("Centroid shape:", centroids.shape) #for debugging purposes

    print("Selecting candidates")
    #get the candidates which are nearest to the centroids
    knn = KNeighborsTimeSeries(n_neighbors=1)
    knn.fit(dataset)
    ind = knn.kneighbors(centroids, return_distance=False)
    print("Neighbors shape", ind.shape) #for debugging purposes

    del dataset

    candidates_name_list = list()
    neighbors_list = list()
    #iterates over first dimension of ind
    print("Selecting neighbors")
    for i in ind:
        neighbor = backup_dataset[i]
        #reshape from (1, sz, d) to (sz, d) and save it to list
        neighbors_list.append(neighbor.reshape(neighbor.shape[1], neighbor.shape[2]))
        del neighbor

    del backup_dataset

    #iterate again over the CSVs to find the candidate nodes name
    #I've decided to do this instead of using a lookup table in order to save up some space
    csvlist = list(csvlist)
    pbar = tqdm(total=len(csvlist))
    i = 0
    print("Finding candidate node names")
    while neighbors_list and i < len(csvlist):
        file_entry = csvlist[i]
        i += 1
        df = pd.read_csv(file_entry, header=0, index_col=0, parse_dates=True, usecols=['Time'] + selectedcols)
        X = df.asfreq(args.freq)
        X = X[selectedcols]
        X = X.to_numpy()
        del df
        for candidate in neighbors_list:
            if np.array_equal(candidate, X):
                candidates_name_list.append(file_entry.stem)
                removearray(neighbors_list, candidate)
                break
        pbar.update(1)
    pbar.close()


        #candidate_name = nodedict[candidate.tobytes()]

    print(candidates_name_list)

    cf = pd.DataFrame(candidates_name_list)

    cf.to_csv(resultsfile)

