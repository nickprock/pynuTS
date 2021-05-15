"""
Created on Wed May 15 2021

@project: pynuTS
@author: nicola procopio
@last_update: 15/05/2021
@description: dimensionality reduction by SAX encoding
@reference: https://iaml.it/blog/serie-storiche-2-sax-encoding
"""

import numpy as np
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
        new_len = int(np.ceil((X_stand.shape[1]/self.windows)))
        #df_PAA = []
        

        return X_stand, new_len
