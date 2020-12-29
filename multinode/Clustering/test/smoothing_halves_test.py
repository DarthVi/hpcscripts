import pandas as pd
from pandas.testing import assert_frame_equal
import pathlib
import unittest

def smooth_data_halves(files, columns, faults, apps):
    cols_to_load = columns + ['faultLabel', 'applicationLabel']

    meanlist = list()
    files_len = len(files)
    
    for i,file in enumerate(files):
        #read only the columns we are interested in
        n = pd.read_csv(file, usecols=cols_to_load, header=0)

        #divide node data in half
        half_len = int(len(n)/2)
        halves = [n.iloc[:half_len, :], n.iloc[half_len:, :]]
        for j,d in enumerate(halves):
            #select only the data relative to the faults we are interested in
            d = d[d['faultLabel'].isin(faults)]
            #select only the data relative to the applications we are interested in
            d = d[d['applicationLabel'].isin(apps)]
            
            #drop the fault label
            d.drop(['faultLabel', 'applicationLabel'], axis=1, inplace=True)

            #reorder columns
            d = d[columns]

            #get the sum of every column, convert Series to dataframe and transpose
            d = d.sum().to_frame().T

            #trick to insert nodename as first column
            d['nodename'] = file.stem + '_' + str(j+1)
            nodename = d.pop('nodename')
            d.insert(0, 'nodename', nodename)
            meanlist.append(d)

    df = pd.concat(meanlist, axis=0).reset_index(drop=True)
    return df

class TestSmoothing(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.filelist = sorted(list(pathlib.Path("./test_toy_data").glob("*.csv")), key=lambda x: int(x.stem[1:]))
        self.columns = ['foo', 'bar', 'zoom']
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def testHealthyAndKripke(self):
        res_df = smooth_data_halves(self.filelist, self.columns, [0], [20])
        df = pd.DataFrame({'nodename': ['N1_1','N1_2', 'N2_1', 'N2_2'], 'foo': [3, 0,  3, 0], 'bar': [30, 0, -3, 0], 'zoom': [-3, 0, 30, 0]})
        self.assertEqual(df, res_df)

    def testHealthyAndKripkeAndAMG(self):
        res_df = smooth_data_halves(self.filelist, self.columns, [0], [20, 21])
        df = pd.DataFrame({'nodename': ['N1_1','N1_2', 'N2_1', 'N2_2'], 'foo': [7, 10, 7, 10], 'bar': [70, 100, -7, -10], 'zoom': [-7, -10, 70, 100]})
        self.assertEqual(df, res_df)

    def testHealthyAndMemeaterAndKripke(self):
        res_df = smooth_data_halves(self.filelist, self.columns, [0, 1], [20])
        df = pd.DataFrame({'nodename': ['N1_1','N1_2', 'N2_1', 'N2_2'], 'foo': [11, 0, 11, 0], 'bar': [110, 0, -11, 0], 'zoom': [-11, 0, 110, 0]})
        self.assertEqual(df, res_df)