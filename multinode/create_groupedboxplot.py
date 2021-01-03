import sys
import pathlib

from utils import scoreboxplot, grouped_scoreboxplot

if __name__ == '__main__':
    
    dfpath = pathlib.Path(str(sys.argv[1]))
    basepath = pathlib.Path(str(sys.argv[2]))
    parentfolder = dfpath.parent
    savefile = parentfolder.joinpath("boxplot_withbaseline.png")
    title = str(sys.argv[3])

    grouped_scoreboxplot(dfpath, savefile, basepath, title)