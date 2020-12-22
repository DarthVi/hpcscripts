import sys
import pathlib

from utils import scoreboxplot

if __name__ == '__main__':
    
    dfpath = pathlib.Path(str(sys.argv[1]))
    parentfolder = dfpath.parent
    savefile = parentfolder.joinpath("boxplot.png")
    title = str(sys.argv[2])

    scoreboxplot(dfpath, savefile, title)