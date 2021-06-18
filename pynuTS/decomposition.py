"""
Created on Wed May 15 2021

@project: pynuTS
@author: nicola procopio
@last_update: 15/05/2021
@description: dimensionality reduction by SAX encoding
@reference: https://iaml.it/blog/serie-storiche-2-sax-encoding
"""

import numpy as np
from pandas import cut
from sklearn.base import BaseEstimator, TransformerMixin

class NaiveSAX(TransformerMixin, BaseEstimator):
    """
    SAX Encoding (Symbolic Aggregate approXimation) is the first symbolic representation for time series that allows for dimensionality reduction and indexing with a lower-bounding distance measure.
    SAX was invented by Eamonn Keogh and Jessica Lin in 2002.

    Parameters
    -----------------------
    levels : list
        default [0.25, 0.75]. Limits in which to fit the values ​​of the series. With default values we have 3 levels. [(min, 1st quartile);(1st quartile, 3rd quartile), (3rd quartile, max)].
    labels: list
        default ["A", "B", "C"]. Labels for SAX Encoding.
    windows : int
        default 2. Time window for PAA (Piecewise Aggregate Approximation).

    Example
    -----------------------
    >> import numpy as np
    >> ts1 = 2.5 * np.random.randn(100,) + 3
    >> ts2 = 4.5 * np.random.randn(100,) + 13
    >> from pynuTS.decomposition import NaiveSAX
    """
    def __init__(self, levels: list = [0.25, 0.75], labels: list = ["A", "B", "C"], windows: int = 2):
        if windows < 2:
            raise ValueError("time window must be at least equal to 2")
        if len(levels) >= len(labels):
            raise ValueError("len(levels) must be equals to len(labels) - 1")
        self.levels=levels
        self.labels=labels
        self.windows=windows
    
    def fit_transform(self, X):
        """
        Parameters
        --------------------
        X: array-like of shape (n_timeSeries, n_timeSteps)

        Return
        --------------------
        SAX_strings: ndarray array of shape (1, n_timeSeries)
            trasformed array. 
        """
        X_mean = np.nanmean(X, axis=1)[np.newaxis]
        X_std = np.nanstd(X, axis=1)[np.newaxis]
        X_stand = (X - X_mean.T)/X_std.T

        # don't lose information

        if (X_stand.shape[1]%self.windows)==0:
            up = int(X_stand.shape[1]/self.windows)
        else:
            up = int((X_stand.shape[1]/self.windows)+1)
        paa = [None]*up
        ind=0
        for i in range(0, X_stand.shape[1], windows):
            avg = np.nanmean(X_stand[:,i:i+windows], axis=1).tolist()
            paa.append(avg)
            ind +=1
        paa = np.transpose(paa)
        """
        binned=[None]*up
        bins = [np.min(paa[j])-.01]
        for j in range(1,len(self.levels)):
            bins.append(np.quantile(paa[j],self.level[0]))
        bins.append(np.max(paa[j])+.01)
        for k in range(len(paa)):
            binned[j] = cut(paa[j], bins, labels=self.labels)
        """
        d = {"vect": X_stand, "up":up, "ind":ind, "paa":paa}

        return d

import numpy as np
ts1 = np.array([[1,3,3,4,5]])
ts2 = np.array([[1,2,3,4]])
ts3 = [ts1,ts2]

z= []
z1= []
sax=NaiveSAX()
for s in ts3:
    x =sax.fit_transform(s)
    z.append(x["ind"])
    z1.append(x["paa"])
print(z)
print(z1)
