from __future__ import annotations
from typing import List
from FileFeatureReader.featurereaders import IFeatureReader
import os

class FeatureReader():
	"""
	Class used to retrieve as a list the most important features saved in a text file.
	It allows the user to specify the strategy used to read the file, thanks to the presence
	of IFeatureReader interface.
	"""

	def __init__(self, reader: IFeatureReader, filepath: str):
		self._filepath = filepath
		self._reader = reader

	@property
	def reader(self) -> IFeatureReader:
		return self._reader

	@property
	def filepath(self) -> str:
		return self._filepath

	@filepath.setter
	def filepath(self, path: str) -> None:
		self._filepath = path
	

	@reader.setter
	def reader(self, reader: IFeatureReader) -> None:
		self._reader = reader

	def getFeats(self) -> List:
		"""
		Get all the features as a list from file
		"""
		try:
			if os.stat(self._filepath).st_size == 0:
				raise ValueError('{} is empty'.format(self._filepath))

			with open(self._filepath) as file:
				try:
					return self._reader.getFeatures(file)
				except ValueError:
					raise
		except IOError:
			raise IOError('{} does not exist'.format(self._filepath))

	def getNFeats(self, n: int) -> List:
		"""
		Get only n most important features from file
		"""
		try:
			if os.stat(self._filepath).st_size == 0:
				raise ValueError('{} is empty'.format(self._filepath))

			with open(self._filepath) as file:
				try:
					return self._reader.getNFeatures(file, n)
				except ValueError:
					raise
		except IOError:
			raise IOError('{} does not exist'.format(self._filepath))
	