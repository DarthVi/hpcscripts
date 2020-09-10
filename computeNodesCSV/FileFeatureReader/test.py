import unittest
from featurereaders import RFEFeatureReader, DTFeatureReader
from featurereader import FeatureReader
from unittest import mock
from unittest.mock import patch
import builtins

class TestFeatureReader(unittest.TestCase):
    def setUp(self):
        self.rfe_feat_reader = FeatureReader(RFEFeatureReader(), "/foo/bar.txt")
        self.dt_feat_reader = FeatureReader(DTFeatureReader(), "/foo/bar.txt")

    def testRFEFull(self):
        feat = ['column1', 'column2', 'column3']
        read_data = 'Header\n---- column1\n---- column2\n---- column3\n'
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mock.Mock()):
            readlist = self.rfe_feat_reader.getFeats()
        self.assertEqual(feat, readlist)

    def testRFEFull2(self):
        feat = ['column1', 'column2']
        read_data = 'Header\n---- column1\n---- column2\n---- column3\n'
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mock.Mock()):
            readlist = self.rfe_feat_reader.getNFeats(2)
        self.assertEqual(feat, readlist)

    def testRFEFull3(self):
        feat = ['column1', 'column2']
        read_data = 'Header\n---- column1\n---- column2\n---- column3\n'
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mock.Mock()):
            with self.assertRaises(ValueError) as context:
                readlist = self.rfe_feat_reader.getNFeats(0)
            self.assertEqual('n parameter is lower than 1 (it is 0)', str(context.exception))

    def testRFEEmptyFile(self):
        mck = mock.Mock()
        attrs = {'st_size': 0}
        mck.configure_mock(**attrs)
        read_data = ''
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mck):
            with self.assertRaises(ValueError) as context:
                readlist = self.rfe_feat_reader.getFeats()
            self.assertEqual('/foo/bar.txt is empty', str(context.exception))

    def testRFEEmptyFile2(self):
        mck = mock.Mock()
        attrs = {'st_size': 0}
        mck.configure_mock(**attrs)
        read_data = ''
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mck):
            with self.assertRaises(ValueError) as context:
                readlist = self.rfe_feat_reader.getNFeats(2)
            self.assertEqual('/foo/bar.txt is empty', str(context.exception))

    def testDTFull(self):
        feat = ['column1', 'column2', 'column3']
        read_data = 'Header\n---- column1: 0.1738919473844908\n---- column2: 0.1738919473844908\n---- column3: 0.1738919473844908\n'
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mock.Mock()):
            readlist = self.dt_feat_reader.getFeats()
        self.assertEqual(feat, readlist)

    def testDTFull2(self):
        feat = ['column1', 'column2']
        read_data = 'Header\n---- column1: 0.1738919473844908\n---- column2: 0.1738919473844908\n---- column3: 0.1738919473844908\n'
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mock.Mock()):
            readlist = self.dt_feat_reader.getNFeats(2)
        self.assertEqual(feat, readlist)

    def testDTFull3(self):
        feat = ['column1', 'column2']
        read_data = 'Header\n---- column1: 0.1738919473844908\n---- column2: 0.1738919473844908\n---- column3: 0.1738919473844908\n'
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mock.Mock()):
            with self.assertRaises(ValueError) as context:
                readlist = self.rfe_feat_reader.getNFeats(0)
            self.assertEqual('n parameter is lower than 1 (it is 0)', str(context.exception))

    def testDTEmpty(self):
        mck = mock.Mock()
        attrs = {'st_size': 0}
        mck.configure_mock(**attrs)
        read_data = ''
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mck):
            with self.assertRaises(ValueError) as context:
                readlist = self.dt_feat_reader.getFeats()
            self.assertEqual('/foo/bar.txt is empty', str(context.exception))

    def testDTEmpty2(self):
        mck = mock.Mock()
        attrs = {'st_size': 0}
        mck.configure_mock(**attrs)
        read_data = ''
        mock_open = mock.mock_open(read_data=read_data)
        with mock.patch("builtins.open", mock_open), mock.patch("os.stat", return_value=mck):
            with self.assertRaises(ValueError) as context:
                readlist = self.dt_feat_reader.getNFeats(2)
            self.assertEqual('/foo/bar.txt is empty', str(context.exception))

    def testRFENotExist(self):
        with self.assertRaises(IOError) as context:
            readlist = self.rfe_feat_reader.getFeats()
        self.assertEqual('/foo/bar.txt does not exist', str(context.exception))

    def testRFENotExist2(self):
        with self.assertRaises(IOError) as context:
            readlist = self.rfe_feat_reader.getNFeats(3)
        self.assertEqual('/foo/bar.txt does not exist', str(context.exception))

    def testDTNotExist(self):
        with self.assertRaises(IOError) as context:
            readlist = self.dt_feat_reader.getFeats()
        self.assertEqual('/foo/bar.txt does not exist', str(context.exception))

    def testDTNotExist(self):
        with self.assertRaises(IOError) as context:
            readlist = self.dt_feat_reader.getNFeats(3)
        self.assertEqual('/foo/bar.txt does not exist', str(context.exception))



if __name__ == '__main__':
    unittest.main