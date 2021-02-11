"""
Created on Thu Jun 18 2020

@project: pynuTS
@author: nicola procopio
@last_update: 11/02/2021
@description: Dynamic Time Warping
@references: https://iaml.it/blog/serie-storiche-3-dynamic-time-warping
"""

def naive_dtw(ts1, ts2, w: int = 1):
    """
    Calculates the distance between two time series using the Dynamic Time Warping

    Parameters
    -----------------------
    ts1, ts2 : 2D numpy array
    w : int.
        default 1. Window parameter

    Returns
    -----------------------
    dist : float.
        The distance between the time series
    DTW_matrix : numpy matrix (or 2D array)
        Distance matrix with warping path.

    Exemple
    -----------------------
    >> import numpy as np
    >> serie_1 = np.array([1, 2, 3, 5, 5, 5, 6], ndmin = 2)
    >> serie_2 = np.array([1, 1, 2, 2, 3, 5], ndmin =2)
    >> from pynuTS.dtw import naive_dtw
    >> dist, DTW_matrix = naive_dtw(ts1 = serie_1, ts2 = serie_2, w=1)
    """

    n = ts1.shape[1]+ 1
    m = ts2.shape[1]+ 1
    DTW = np.zeros([n,m])
    w = max([w, abs(n-m)])
    for i in range(1,n):
        for j in range(max([1,i-w]), min([m, i+w])):
            dist = abs(ts1[:,i-1] - ts2[:,j-1]) 
            DTW[i,j] = dist + min([DTW[i-1,j], DTW[i,j-1], DTW[i-1,j-1]])
    return DTW[-1,-1], DTW
