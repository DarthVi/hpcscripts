import sys
import pathlib

from utils import clustering_multirun_overall_boxplot

if __name__ == '__main__':
    
    dfpath = pathlib.Path(str(sys.argv[1]))
    randompath = pathlib.Path(str(sys.argv[2]))
    num_train = int(sys.argv[3])
    total_nodes = int(sys.argv[4])
    parentfolder = dfpath.parent
    savefile = parentfolder.joinpath("boxplot_clustering_multirunrandom_mean.png")
    title = str(sys.argv[5])

    clustering_multirun_overall_boxplot(dfpath, savefile, randompath, num_train, total_nodes, title)