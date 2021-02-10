"""
Created on Thu Jun 18 2020

@project: pynuTS
@author: nicola procopio
@last_update: 09/02/2021
@description: Time Series Clustering
@references: https://iaml.it/blog/serie-storiche-3-dynamic-time-warpi
"""

from numpy import array
from tqdm import tqdm
from dtw import accelerated_dtw
import random

class DTWKmeans:
    """
    K - Means clustering algorithm using DTW for misure similarity.

    Parameters
    -----------------------
    num_clust : int
        number of cluster.
    num_iter : int
        default 1. Number of iterations.
    w :  int.
        default 1. Window parameter
    euclidean : bool.
        default True. If True compute DTW with euclidean distance, else use the cosine similarity.

    Example
    -----------------------
    >> import numpy as np
    >> import pandas as pd
    >> ts1 = 2.5 * np.random.randn(100,) + 3
    >> X_1 = pd.Series(ts1)
    >> ts2 = 2 * np.random.randn(100,) + 5
    >> X_2 = pd.Series(ts2)
    >> ts3 = -2.5 * np.random.randn(100,) + 3
    >> X_3 = pd.Series(ts3)
    >> list_of_series = [X_1, X_2, X_3]
    >> from pynuTS.clustering import DTWKmeans
    >> clts = DTWKmeans(num_clust = 2, num_iter = 5)
    >> clts.fit(list_of_series)
    >> ts4 = 3.5 * np.random.randn(100,) + 2
    >> ts5 = -3.5 * np.random.randn(100,) + 2
    >> X_4 = pd.Series(ts4)
    >> X_5 = pd.Series(ts5)
    >> list_new = [X_4, X_5]
    >> clts.predict(list_new)
    """
    def __init__(self, num_clust : int, num_iter : int = 1, w: int = 1, euclidean: bool = True):
        if num_clust < 1:
            raise ValueError("number of cluster must be at least equal to 1")
        if num_iter < 1:
            raise ValueError("number of iteration must be at least equal to 1")
        if w < 1:
            raise ValueError("window parameter must be at least equal to 1")

        self.num_clust = num_clust
        self.num_iter = num_iter
        self.w = w
        self.euclidean = euclidean
    
    def fit(self, data: list):
        """
        Compute k-means clustering.

        Parameters
        -----------------------
        data : a list of pandas Series
        """

        centroids = random.sample(data,self.num_clust)
        for _ in tqdm(range(self.num_iter)):
            assignments={}
            for e in range(len(centroids)):
                assignments.update({e:[]})
            for ind,i in  enumerate(data):
                min_dist = float('inf')
                closest_clust = None
                for c_ind,j in enumerate(centroids):
                    if self.euclidean:
                        criterio = 'euclidean'
                    else:
                        criterio = 'cosine'
                    fastDTW, _, _, _ = accelerated_dtw(array(i), array(j), dist=criterio, warp=self.w)
                    if fastDTW<=min_dist:
                        min_dist = fastDTW
                        closest_clust = c_ind
                if closest_clust in assignments:
                    assignments[closest_clust].append(ind)
            for key in assignments:
                clust_sum=0
                for k in assignments[key]:
                    clust_sum=clust_sum+data[k]
                if len(assignments[key])>0:
                    centroids[key]= clust_sum/len(assignments[key])
        self.cluster_centers_, self.labels_ = centroids, assignments
        return self

    def predict(self, data: list):
        """
        Assingn new series based on precalculated centroid.

        Parameters
        -----------------------
        data : a list of pandas Series

        Returns
        -----------------------
        assignments: a dictionary {cluster: index_series}
        """

        assignments_new={}

        for e in tqdm(range(len(self.centroids))):
            assignments_new.update({e:[]})
        for ind,i in  enumerate(data):
            dist = []
            for _, j in enumerate(self.centroids):
                if self.euclidean:
                    criterio = 'euclidean'
                else:
                    criterio = 'cosine'
                fastDTW, _, _, _ = accelerated_dtw(array(i), array(j), dist=criterio, warp=self.w)
                dist.append(fastDTW)
            clust = dist.index(min(dist))
            assignments_new[clust].append(ind)
        return assignments_new