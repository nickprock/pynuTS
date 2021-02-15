# embryo of unit test suite for pynuTS clustering

import pytest
from pynuTS.clustering import DTWKmeans
import numpy as np
import pandas as pd


class TestDTWKmeans(object):

    def test_example(self):
        num_clusters = 2
        iterations = 5
        ts1 = 2.5 * np.random.randn(100,) + 3
        X_1 = pd.Series(ts1)
        ts2 = 2 * np.random.randn(100,) + 5
        X_2 = pd.Series(ts2)
        ts3 = -2.5 * np.random.randn(100,) + 3
        X_3 = pd.Series(ts3)
        list_of_series = [X_1, X_2, X_3]
        from pynuTS.clustering import DTWKmeans
        clts = DTWKmeans(num_clust = num_clusters, num_iter = iterations)
        clts.fit(list_of_series)
        ts4 = 3.5 * np.random.randn(100,) + 2
        ts5 = -3.5 * np.random.randn(100,) + 2
        X_4 = pd.Series(ts4)
        X_5 = pd.Series(ts5)
        list_new = [X_4, X_5]
        clustering_dict = clts.predict(list_new)

        assert type(clustering_dict) is dict
        assert len(clustering_dict) == num_clusters