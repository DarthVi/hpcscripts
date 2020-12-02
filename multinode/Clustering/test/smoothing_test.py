import pandas as pd
from pandas.testing import assert_frame_equal
import pathlib
import unittest

def smooth_data(files, columns, faults, apps):
    #my_bar = st.progress(0)
    cols_to_load = columns + ['faultLabel', 'applicationLabel']

    meanlist = list()
    #perc_upgrade = int(100 / length)
    
    
    for file in files:
        #print(file)
        #read only the columns we are interested in
        n = pd.read_csv(file, usecols=cols_to_load, header=0)
        #select only the data relative to the faults we are interested in
        n = n[n['faultLabel'].isin(faults)]
        #select only the data relative to the applications we are interested in
        n = n[n['applicationLabel'].isin(apps)]
        #print(n)
        #drop the fault label
        n.drop(['faultLabel', 'applicationLabel'], axis=1, inplace=True)

        #reorder columns
        n = n[columns]

        #get the mean of every column, convert Series to dataframe and transpose
        #print(n.mean())
        n = n.sum().to_frame().T
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
        res_df = smooth_data(self.filelist, self.columns, [0], [20])
        df = pd.DataFrame({'nodename': ['N1', 'N2'], 'foo': [3, 3], 'bar': [30, -3], 'zoom': [-3, 30]})
        self.assertEqual(df, res_df)

    def testHealthyAndKripkeAndAMG(self):
        res_df = smooth_data(self.filelist, self.columns, [0], [20, 21])
        df = pd.DataFrame({'nodename': ['N1', 'N2'], 'foo': [17, 17], 'bar': [170, -17], 'zoom': [-17, 170]})
        self.assertEqual(df, res_df)

    def testHealthyAndMemeaterAndKripke(self):
        res_df = smooth_data(self.filelist, self.columns, [0, 1], [20])
        df = pd.DataFrame({'nodename': ['N1', 'N2'], 'foo': [11, 11], 'bar': [110, -11], 'zoom': [-11, 110]})
        self.assertEqual(df, res_df)