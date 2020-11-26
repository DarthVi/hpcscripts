import argparse
import pathlib

from utils import summaryboxplot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path in which there are the CSV to plot and in which to save the plot images")
    args = parser.parse_args()

    path = pathlib.Path(args.path)

    faults = ['overall','healthy', 'memeater','memleak', 'membw', 'cpuoccupy','cachecopy','iometadata','iobandwidth']

    for fault in faults:
        summaryboxplot(path, fault, path, 'num_train', fault + " F1-scores")
