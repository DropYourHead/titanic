import pandas as pd
import numpy

class DNode:
    def __init__(self, feature=None, val=None):
        #feature : the feature we choose to split, a string, leaf node if feature is None
        #value : the value in the node, dataframe
        self.value = val
        self.feature = feature
        self.pList = {}
        self.valueDiv(feature, val)
        
    def valueDiv(self, feature, val):
        div = val.groupby(feature)
        for k, v in div:
            self.pList[k] = DNode(feature=None, val=v)
        
            


class DTree:
    def __init__(self, maxDepth, threshold, minNum, val):
        self.root = None
        self.maxDepth = maxDepth
        self.threshold = threshold
        self.minNum = minNum
        self.val = val

    def calcEntropy(self, features, val):

