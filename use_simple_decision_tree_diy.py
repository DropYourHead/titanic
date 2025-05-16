import pandas as pd
import numpy as np

class DNode:
    def __init__(self, feature=None, X=None, y=None):
        #feature : the feature we choose to split, a string, leaf node if feature is None
        #value : the value in the node, dataframe
        self.X = X
        self.y = y
        self.feature = feature
        self.pList = {}
        self.valueDiv(feature, X)
        
    def valueDiv(self, feature, X):
        div = X.groupby(feature)
        for k, v in div:
            self.pList[k] = DNode(feature=None, val=v)
        
            


class DTree:
    def __init__(self, maxDepth, threshold, minNum):
        self.root = None
        self.maxDepth = maxDepth
        self.threshold = threshold
        self.minNum = minNum

    def calcEntropy(self, features, val):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass