from __future__ import annotations
from abc import ABC, abstractmethod
from io import TextIOWrapper
import re
from typing import List

class IFeatureReader(ABC):

    @abstractmethod
    def getFeatures(self, file: TextIOWrapper):
        pass

    @abstractmethod
    def getNFeatures(self, file: TextIOWrapper, n: int):
        pass


class RFEFeatureReader(IFeatureReader):

    def getFeatures(self, file: TextIOWrapper) -> List:
        featlist = []
        next(file) #skip first line
        for line in file:
            sline = re.sub(r"---- ", "", line).strip()
            featlist.append(sline)
        return featlist

    def getNFeatures(self, file: TextIOWrapper, n: int) -> List:
        featlist = []
        if n < 1:
            raise ValueError('n parameter is lower than 1 (it is {})'.format(n))
        next(file) #skip first line
        for i in range(n):
            line = file.readline().strip()
            sline = re.sub(r"---- ", "", line)
            featlist.append(sline)
        return featlist

class DTFeatureReader(IFeatureReader):

    def getFeatures(self, file: TextIOWrapper) -> List:
        featlist = []
        next(file) #skip first line
        for line in file:
            sline = re.sub(r"---- ", "", line).strip()
            sline = re.sub(r":.*", "", sline)
            featlist.append(sline)
        return featlist

    def getNFeatures(self, file: TextIOWrapper, n: int) -> List:
        featlist = []
        if n < 1:
            raise ValueError('n parameter is lower than 1 (it is {})'.format(n))
        next(file) #skip first line
        for i in range(n):
            line = file.readline().strip()
            sline = re.sub(r"---- ", "", line)
            sline = re.sub(r":.*", "", sline)
            featlist.append(sline)
        return featlist
