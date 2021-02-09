"""
Created on Wed Apr 22 2020

@project: pynuTS
@author: nicola procopio
@last_update: 09/02/2021
@description: Outlier Detection with SAX encoding
@reference: https://iaml.it/blog/serie-storiche-2-sax-encoding
"""

from pandas import DataFrame, cut

class SAXEncoding:
    """
    SAX encoding is a method used to simplify time series through the summarization of time intervals, wanting to find anomalous patterns.

    Parameters
    -----------------------
    windows : int
        default 2. Time window for PAA (Piecewise Aggregate Approximation).
    outlier_freq : int
        default 1. Maximum number of time series in a pattern for it to be considered an outlier.
    
    Example
    -----------------------
    >> import numpy as np
    >> import pandas as pd
    >> ts1 = 2.5 * np.random.randn(100,) + 3
    >> ts2 = 4.5 * np.random.randn(100,) + 13
    >> ts3 = -2.5 * np.random.randn(100,) + 3
    >> ts4 = -5 * np.random.randn(100,) -2.3333
    >> X = pd.DataFrame([ts1, ts1, ts2, ts3, ts4]).T
    >> from pynuTS.anomaly_detection import SAXEncoding
    >> sax = SAXEncoding()
    >> df, binned, freq, dictionary = sax.fit_transform(X)
    """
    def __init__(self, windows : int = 2, outlier_freq : int = 1):
        if windows < 2:
            raise ValueError("time window must be at least equal to 2")
        if outlier_freq < 1:
            raise ValueError("outlier frequency must be at least equal to 1")

        self.windows = windows
        self.outlier_freq = outlier_freq

    def fit_transform(self, data_frame):
        """
        Fit of parameters on the training set X, but it also returns a transformed X'

        Parameters
        -----------------------
        data_frame : a data frame. Each column is a time series

        Returns
        ----------------------
        df : pandas DataFrame.
            the input data_frame traslate with an additional boolean column, the time series is an outlier or not.
        binned : pandas DataFrame.
            a data_frame with the SAX strings.
        freq : pandas DataFrame. 
            frequency for each pattern
        dictionary : a dict.
            a dict with:
                key -> outlier/standard series
                values -> the colnames of input data_frame
        """
        df = data_frame.T
        df.index = range(0, df.shape[0])
        df.mean(axis = 1, skipna = True)
        df.std(axis = 1, skipna = True)
        df_stand = ((df.T - df.mean(axis=1))/df.std(axis=1)).T
        if (df_stand.shape[1]%self.windows)==0:
            up = int(df_stand.shape[1]/self.windows)
        else:
            up = int((df_stand.shape[1]/self.windows)+1)
        df_PAA = DataFrame(index = range(0, df_stand.shape[0]), columns = range(0, up))
        ind = 0
        for i in range(0,df_stand.shape[1], self.windows):
            avg = df_stand.iloc[:,i:i+self.windows].mean(axis=1)
            df_PAA[ind] = avg.values
            ind +=1
        binned = DataFrame(index = df_PAA.index, columns = df_PAA.columns)
        for j in range(0, df_PAA.shape[1]):
            bins = []
            bins.append(df_PAA[j].min()-.01)
            bins.append(df_PAA[j].quantile([0.25]).values[0])
            bins.append(df_PAA[j].quantile([0.75]).values[0])
            bins.append(df_PAA[j].max()+.01)
            labels = ["A", "B", "C"]
            binned[j] = cut(df_PAA[j], bins, labels=labels)
        binned['sequence'] = binned.apply(''.join, axis=1)
        freq = binned.sequence.value_counts()
        df['outlier'] = binned['sequence'].isin(list(freq[freq<=self.outlier_freq].index))
        idx_true = df[df['outlier']==True].index
        idx_false = df[df['outlier']==False].index
        dictionary = dict({'outlier_TS':data_frame.columns[idx_true].tolist(), 'standard_TS':data_frame.columns[idx_false].to_list()})
        return df, binned, freq, dictionary