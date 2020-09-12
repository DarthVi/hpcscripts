"""
2020-09-12 14:09

@author: Vito Vincenzo Covella
"""

import numpy as np
import pandas as pd
import sys

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Please insert a file to subsample and a label to subsample")
        exit(1)

    #input csv file
    input_file = str(sys.argv[1])
    input_name = input_file.split('.')[0]
    nodename = input_file.split('_')[0].split('/')[-1]

    #input label to subsample
    labelToSubsample = int(sys.argv[2])

    #sanity check for the input label to subsample
    if labelToSubsample < 0 or labelToSubsample > 7:
        print("Wrong label given as input")
        exit(1)

    df = pd.read_csv(input_file, header = 0)

    counts = df['label'].value_counts(sort=False)

    #take all the counts except the count of the label we want to subsample and take the mean of this new list
    countX = list(filter(lambda x: x != counts[labelToSubsample], counts))
    targetValue = int(np.mean(countX))

    #take a subsample of the label selected
    df_label = df[df['label'] == labelToSubsample].sample(targetValue, random_state=42)
    #concatenate the subsampled dataframe with the rest of the dataframe
    df = pd.concat([df_label, df[df['label'] != labelToSubsample]], axis = 0)

    substring = "_subsampled" + str(labelToSubsample) + ".csv"
    #save to file the new subsampled dataframe
    df.to_csv(input_name + substring, index=False, header=True)