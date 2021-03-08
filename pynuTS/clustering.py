"""
Created on Thu Jun 18 2020

@project: pynuTS
@author: nicola procopio
@last_update: 09/02/2021
@description: Time Series Clustering
@references: https://iaml.it/blog/serie-storiche-3-dynamic-time-warping
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
        default 1. Max number of iterations
    num_init : int
        default 1. Number of different initializations
    w :  int.
        default 1. Window parameter
    euclidean : bool.
        default True. If True compute DTW with euclidean distance, else use the cosine similarity.
    random_seed : None or any  type suitable for random seed initialization (usually int) 
        default None. Random seed initialization for reproduceability, not initialized if None

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
    def __init__(self, num_clust : int, num_iter : int = 1, num_init = 1,
                       w: int = 1, euclidean: bool = True, random_seed = None):
        if num_clust < 1:
            raise ValueError("number of cluster must be at least equal to 1")
        if num_iter < 1:
            raise ValueError("number of iteration must be at least equal to 1")
        if num_init < 1:
            raise ValueError("number of initializations must be at least equal to 1")
        if w < 1:
            raise ValueError("window parameter must be at least equal to 1")

        self.num_clust = num_clust
        self.num_iter = num_iter
        self.num_init = num_init
        self.w = w
        self.criterion = {True : 'euclidean', False : 'cosine'}[euclidean]
        self.seed = random_seed
        if not self.seed is None :
            random.seed(self.seed)
    
    def fit(self, data: list, patience: int = 5):
        """
        Compute k-means clustering.

        Parameters
        -----------------------
        data : a list of pandas Series
        patience: int. 
            default 5. number of iterations with no improvement after which training will be stopped.
        """

        min_inertia = float('inf')
        for init_run in range(self.num_init):
            centroids = self._init_centroids(data)
            stable_count = 0
            old_assignments = {}
            for iter_run in tqdm(range(self.num_iter)):
                assignments,centroids = self._kmeans_iteration(data,centroids)
                stable_count = _increment_or_reset(stable_count,assignments,old_assignments)
                if stable_count >= patience :
                    break
                old_assignments = assignments
            if (inertia := self._generalized_inertia(centroids, assignments, data)) < min_inertia :
                self.cluster_centers_, self.labels_ = centroids, assignments
                min_inertia = inertia
        return self

    def _init_centroids(self,data):
        """Initialize centroids of self sampling from data, with random seed if specified
        Parameters 
        -----------------------
        data : a list of pandas Series

        Returns
        -----------------------
        centroids : a list of pandas series
        """


        centroids = random.sample(data,self.num_clust)
        return centroids

    def _kmeans_iteration(self,data,centroids):
        """A single iteration of k-means lloyd.
    
        Parameters
        ----------
        data : a list of pandas Series

        centroids : the current centroids as list of pandas Series, as many as self.num_clust

        Returns
        -----------------------
        assignements : the current samples assignements as dictionary in the form { e : [index] } 
                       where e is the centroid number and the indexes in the list are the indexes 
                       of the data elements in the relevent centroid 
                                
        """
        # compute assignements
        assignments={ e : [] for e in range(self.num_clust) } 
        for ind,i in  enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind,j in enumerate(centroids):
                fastDTW, _, _, _ = accelerated_dtw(array(i), array(j), dist=self.criterion, warp=self.w)
                if fastDTW<=min_dist:
                    min_dist = fastDTW
                    closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
        # update centroids
        new_centroids = centroids.copy()
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            if len(assignments[key])>0:
                new_centroids[key]= clust_sum/len(assignments[key])

        return assignments,new_centroids

    def _inertia(self, data : list):
        """
        Compute inertia of clusterization given the current centroids. 
        inertia = sum of squared distances of cluster members to cluster centroids
        see definition https://scikit-learn.org/stable/modules/clustering.html#k-means

        Parameters
        -----------------------
        data : a list of pandas Series

        Returns
        -----------------------
        intertia : float 
        """
        return self._generalized_inertia(self.cluster_centers_, self.labels_, data)

    def _generalized_inertia(self, centroids, labels, data):
        inertia = 0
        for e,centroid in enumerate(centroids):
            members = labels[e]
            for member_index in members:
                i = centroid
                j = data[member_index]
                fastDTW, _, _, _ = accelerated_dtw(i.values, j.values, dist=self.criterion, warp=self.w)
                inertia += fastDTW ** 2
        return inertia
    

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

        for e in tqdm(range(len(self.cluster_centers_))):
            assignments_new.update({e:[]})
        for ind,i in  enumerate(data):
            dist = []
            for _, j in enumerate(self.cluster_centers_):
                fastDTW, _, _, _ = accelerated_dtw(array(i), array(j), dist=self.criterion, warp=self.w)
                dist.append(fastDTW)
            clust = dist.index(min(dist))
            assignments_new[clust].append(ind)
        return assignments_new        

def _increment_or_reset(counter,new,old):
    if new == old :
        return counter + 1
    else :
        return 0
