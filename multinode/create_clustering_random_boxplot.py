import sys
import pathlib

from utils import clustering_grouped_scoreboxplot

if __name__ == '__main__':
    
    dfpath = pathlib.Path(str(sys.argv[1]))
    basepath = pathlib.Path(str(sys.argv[2]))
    randompath = pathlib.Path(str(sys.argv[3]))
    parentfolder = dfpath.parent
    savefile = parentfolder.joinpath("boxplot_clustering_multirunrandom_withbaseline_mean.png")
    title = str(sys.argv[4])

    clustering_grouped_scoreboxplot(dfpath, savefile, basepath, randompath, title)