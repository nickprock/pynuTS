"""
Created on Thu Jun 18 2020

@project: pynuTS
@author: nicola procopio
@last_update: 03/08/2026
@description: Time Series Clustering
@references: https://iaml.it/blog/serie-storiche-3-dynamic-time-warping
"""

import random

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from .dtw import dtw_distance


def _as_array(ts):
    """Return a time series as a numpy array, whatever container it comes in."""
    values = getattr(ts, "values", ts)
    return np.asarray(values, dtype=float)


def _barycenter(members):
    """
    Average a group of time series element-wise.

    Series shorter than the longest one are padded with NaN and ignored
    position by position, so that series of different lengths - the very reason
    one uses DTW in the first place - do not silently collapse into NaN.

    Note that this is the plain Euclidean mean, *not* a DTW barycenter: the
    methodologically correct centroid for DTW k-means is DBA (Petitjean et al.,
    2011) or the soft-DTW barycenter (Cuturi & Blondel, 2017). This is a known
    approximation, see the README.

    Parameters
    -----------------------
    members : a non empty list of array-like

    Returns
    -----------------------
    centroid : 1D numpy array
    """
    arrays = [_as_array(m) for m in members]
    length = max(a.shape[0] for a in arrays)
    padded = np.full((len(arrays), length), np.nan)
    for row, a in enumerate(arrays):
        padded[row, :a.shape[0]] = a
    return np.nanmean(padded, axis=0)


class DTWKmeans(BaseEstimator):
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
        default 1. Window parameter, the half-width of the Sakoe-Chiba band
        constraining the warping path.
    criterion : str.
        default 'euclidean'. DTWKMeans support two kind of distance 'euclidean' and 'cosine'.
        Note that 'cosine' is only meaningful on multivariate series: on
        univariate ones it degenerates to a sign comparison.
    seed : None or any  type suitable for random seed initialization (usually int)
        default None. Random seed initialization for reproduceability, not initialized if None

    Attributes
    -----------------------
    cluster_centers_ : list of 1D numpy array
        available after fit.
    labels_ : dict {cluster: [index of the series]}
        available after fit.
    inertia_ : float
        sum of squared DTW distances of the members to their centroid,
        available after fit.

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
                       w: int = 1, criterion: str = 'euclidean', seed = None):
        if num_clust < 1:
            raise ValueError("number of cluster must be at least equal to 1")
        if num_iter < 1:
            raise ValueError("number of iteration must be at least equal to 1")
        if num_init < 1:
            raise ValueError("number of initializations must be at least equal to 1")
        if w < 1:
            raise ValueError("window parameter must be at least equal to 1")
        if criterion not in ["euclidean", "cosine"]:
            raise ValueError("DTWKMeans support only two kind of distance 'euclidean' and 'cosine'")

        self.num_clust = num_clust
        self.num_iter = num_iter
        self.num_init = num_init
        self.w = w
        self.criterion = criterion
        self.seed = seed

    def _distance(self, ts1, ts2):
        return dtw_distance(_as_array(ts1), _as_array(ts2), w=self.w, criterion=self.criterion)

    def fit(self, data: list, patience: int = 5):
        """
        Compute k-means clustering.

        Parameters
        -----------------------
        data : a list of pandas Series
        patience: int.
            default 5. number of iterations with no improvement after which training will be stopped.
        """
        data = list(data)
        if len(data) < self.num_clust:
            raise ValueError(
                "number of series (%d) must be at least equal to the number of "
                "clusters (%d)" % (len(data), self.num_clust)
            )

        # a local generator keeps the global random state untouched, so seeding
        # DTWKmeans no longer leaks into the rest of the caller's program
        rng = random.Random(self.seed)

        min_inertia = float('inf')
        for init_run in range(self.num_init):
            centroids = self._init_centroids(data, rng)
            stable_count = 0
            old_assignments = {}
            for iter_run in tqdm(range(self.num_iter)):
                assignments, centroids = self._kmeans_iteration(data, centroids)
                stable_count = _increment_or_reset(stable_count, assignments, old_assignments)
                if stable_count >= patience:
                    break
                old_assignments = assignments
            if (inertia := self._generalized_inertia(centroids, assignments, data)) < min_inertia:
                self.cluster_centers_, self.labels_ = centroids, assignments
                min_inertia = inertia
        self.inertia_ = min_inertia
        return self

    def _init_centroids(self, data, rng):
        """Initialize centroids of self sampling from data, with random seed if specified

        Parameters
        -----------------------
        data : a list of pandas Series
        rng : a random.Random instance

        Returns
        -----------------------
        centroids : a list of 1D numpy array
        """
        return [_as_array(c) for c in rng.sample(data, self.num_clust)]

    def _kmeans_iteration(self, data, centroids):
        """A single iteration of k-means lloyd.

        Parameters
        ----------
        data : a list of pandas Series

        centroids : the current centroids as list of 1D numpy array, as many as self.num_clust

        Returns
        -----------------------
        assignements : the current samples assignements as dictionary in the form { e : [index] }
                       where e is the centroid number and the indexes in the list are the indexes
                       of the data elements in the relevent centroid

        """
        # compute assignements
        assignments = {e: [] for e in range(self.num_clust)}
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = 0
            for c_ind, j in enumerate(centroids):
                distance = self._distance(i, j)
                if distance < min_dist:
                    min_dist = distance
                    closest_clust = c_ind
            assignments[closest_clust].append(ind)
        # update centroids: empty clusters keep their previous centroid
        new_centroids = list(centroids)
        for key, members in assignments.items():
            if len(members) > 0:
                new_centroids[key] = _barycenter([data[k] for k in members])

        return assignments, new_centroids

    def _inertia(self, data: list):
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
        self._check_is_fitted()
        return self._generalized_inertia(self.cluster_centers_, self.labels_, data)

    def _generalized_inertia(self, centroids, labels, data):
        inertia = 0
        for e, centroid in enumerate(centroids):
            for member_index in labels[e]:
                inertia += self._distance(centroid, data[member_index]) ** 2
        return inertia

    def _check_is_fitted(self):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("this DTWKmeans instance is not fitted yet, call 'fit' first")

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
        self._check_is_fitted()

        assignments_new = {e: [] for e in range(len(self.cluster_centers_))}
        for ind, i in tqdm(enumerate(data), total=len(data)):
            dist = [self._distance(i, j) for j in self.cluster_centers_]
            assignments_new[dist.index(min(dist))].append(ind)
        return assignments_new


def _increment_or_reset(counter, new, old):
    if new == old:
        return counter + 1
    else:
        return 0
