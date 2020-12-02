import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import BayesianGaussianMixture
from sklearn.datasets import make_blobs
import plotly.express as px
from clusteringApp import compute_BGMM

class TestBGMM(unittest.TestCase):
	@classmethod
	def setUpClass(cls)	:
		super(TestBGMM, cls).setUpClass()
		#create some toy data with 4 clusters
		cls.X, cls.y = make_blobs(n_samples=400, centers=4, n_features=3, cluster_std=0.60, random_state=0)
		#convert to dataframe format because the tested function accepts a dataframe with nodename as column
		cls.df = pd.DataFrame({'nodename': list(range(0, len(cls.X))), 'foo': cls.X[:, 0], 'bar': cls.X[:, 1], 'zoom': cls.X[:, 2]})
		#get how many unique clusters there are
		original_clusters = np.unique(cls.y)
		#group the data in the same "sublist" if it belongs to the same A PRIORI KNOWN cluster
		cls.grouped = [[x.tolist() for x,y in zip(cls.X, cls.y) if y == cluster] for cluster in original_clusters]

	def testCompute_BGMM(self):
		res_df = compute_BGMM(TestBGMM.df, 10, 'foo', 'bar', 'zoom')
		# fig = px.scatter_3d(res_df, x='foo', y='bar', z='zoom', color='cluster', hover_name='nodename', hover_data=['distance'])
		# fig.show()
		
		pred_y = res_df['cluster'].astype(int).to_numpy()
		res_X = res_df.drop(['nodename', 'distance', 'cluster'], axis=1).to_numpy()
		pred_clusters = np.unique(pred_y)
		
		#group the data in the same sublist if it belongs to the same PREDICTED cluster
		grouped_pred = [[x.tolist() for x,y in zip(res_X, pred_y) if y == cluster] for cluster in pred_clusters]

		#for each PREDICTED group check if there's an EQUAL GROUP in the A PRIORI list of grouped data
		for g_pred in grouped_pred:
			self.assertTrue(g_pred in TestBGMM.grouped)
