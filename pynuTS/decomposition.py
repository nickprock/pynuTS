"""
Created on Wed May 15 2021

@project: pynuTS
@author: nicola procopio
@last_update: 07/07/2021
@description: dimensionality reduction by SAX encoding
@reference: https://iaml.it/blog/serie-storiche-2-sax-encoding
"""

from numpy.core.records import array
from numpy.lib.function_base import quantile
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from pandas import Series

class NaiveSAX(BaseEstimator, TransformerMixin):
    def __init__(self, levels: list = ["A", "B", "C"], bounds: list = [0.25, 0.75], windows: int = 2, quantile: bool = True):
        """
        SAX Encoding (Symbolic Aggregate approXimation) is the first symbolic representation for time series that allows for dimensionality reduction and indexing with a lower-bounding distance measure.
        SAX was invented by Eamonn Keogh and Jessica Lin in 2002.

        Parameters
        -----------------------
        bounds : list
            default [0.25, 0.75]. Limits in which to fit the values of the series. With default values we have 3 levels. [(min, 1st quartile);(1st quartile, 3rd quartile), (3rd quartile, max)].
        levels: list
            default ["A", "B", "C"]. Labels for SAX Encoding.
        windows : int
            default 2. Time window for PAA (Piecewise Aggregate Approximation).
        quantile: bool
            default True. If False the values in bounds are used without apply any function.

        Returns
        -----------------------
        sax_string

        Example
        -----------------------
        >> import numpy as np
        >> ts1 = 2.5 * np.random.randn(100,) + 3
        >> ts2 = 4.5 * np.random.randn(100,) + 13
        >> from pynuTS.decomposition import NaiveSAX
        >> sax = NaiveSAX()
        >> ts1_decomposed = sax.fit_transform(ts1)
        >> print(ts1_decomposed)
        >> ts3 = np.vstack((ts1, ts2))
        >> ts3_decomposed = np.apply_along_axis(sax.fit_transform, 1, ts3)
        >> print(ts3_decomposed)
        """
        try:
            if len(levels)!= (len(bounds) + 1):
                raise ValueError("Length of levels must be equals at length of bounds plus 1")
        except ValueError as error:
            print('Caught an error: ' + repr(error))
        
        try:
            if windows<1:
                raise ValueError("Windows must be a positive integer")
        except ValueError as error:
            print('Caught an error: ' + repr(error))

        self.windows = windows
        self.bounds = bounds
        self.levels = levels
        self.quantile = quantile

        
    def fit_transform(self, X):
        """
        Parameters
        --------------------
        X: array-like of shape (1, n_timeSteps)

        Return
        --------------------
        SAX_strings: ndarray array of shape (1, n_timeSeries)
            trasformed array. 
        """

        try:
            if isinstance(X, list):
                X = np.array(X)
            elif isinstance(X, Series):
                X = X.values
            elif isinstance(X, np.ndarray):
                pass
            else:
                raise TypeError("X must be a numpy.array or a list or a pandas Series")
        except TypeError as error:
            print('Caught an error: ' + repr(error))
        
        try:
            if X.ndim > 1:
                raise TypeError("X must be a 1-D numpy.array")
        except TypeError as error:
            print('Caught an error: ' + repr(error))
            
        
        if (len(X)%self.windows)==0:
            up = int(len(X)/self.windows)
        else:
            up = int((len(X)/self.windows)+1)
        #print(up)
    
        df_PAA = []
    
        for i in range(0,len(X), self.windows):
            avg = np.nanmean(X[i:i+self.windows])
            #print(avg)
            df_PAA.append(avg)
        
        binned = []

        if self.quantile:
            for j in range(len(df_PAA)):
                if df_PAA[j] < np.quantile(df_PAA,self.bounds[0]):
                    binned.append(self.levels[0])
                for k in range(1, len(self.bounds)):
                    if ((df_PAA[j] >= np.quantile(df_PAA,self.bounds[k-1])) & (df_PAA[j] < np.quantile(df_PAA,self.bounds[k]))):
                        binned.append(self.levels[k])
                if df_PAA[j] >= np.quantile(df_PAA,self.bounds[-1]):
                    binned.append(self.levels[-1])
        else:
            for j in range(len(df_PAA)):
                if df_PAA[j] < self.bounds[0]:
                    binned.append(self.levels[0])
                for k in range(1, len(self.bounds)):
                    if ((df_PAA[j] >= self.bounds[k-1]) & (df_PAA[j] < self.bounds[k])):
                        binned.append(self.levels[k])
                if df_PAA[j] >= self.bounds[-1]:
                    binned.append(self.levels[-1])
        
        sax_string = ''.join(binned)
        return sax_string