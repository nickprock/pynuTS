# embryo of unit test suite for pynuTS clustering

import pytest
from pynuTS.clustering import DTWKmeans
import numpy as np
import pandas as pd

from demos.ts_gen import make_slopes_dataset,make_flat_dataset

class TestDTWKmeans_end2end(object):
    def test_example(self):
        """Example of clustering usage as defined in the docstring of DTWKmeans class""" 
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

    def test_centroids_fitting_data_simple_slopes(self):
        """Example of clustering with simple slopes series
        After fit, centroids values shall match data values
        """ 
        slopes = [0.3,0,-0.3]
        list_of_series = make_slopes_dataset(slopes,10,additive_noise_factor=0.0,intercept_noise_factor=0.0,lengths=[3])
        # setting random seed othewise fitting may end with two matching centroids 
        clts = DTWKmeans(num_clust = 3, num_iter = 10, w=1,euclidean=True,random_seed=101)

        clts.fit(list_of_series)

        df_data = pd.DataFrame(list_of_series).drop_duplicates().sort_values(by=2)
        df_centroids = pd.DataFrame(clts.cluster_centers_).drop_duplicates().sort_values(by=2)
        assert np.allclose(df_data,df_centroids)

class TestDTWKmeans_init(object):
    def test_DTWKmeans_init_default_kwargs(self):
        num_clusters = 2
        clts = DTWKmeans(num_clust = num_clusters)
        assert clts

    @pytest.mark.parametrize("num_clusters", [0, -1, 1, 5, 100]) 
    def test_DTWKmeans_init_clusters_acceptance_range(self,num_clusters):
        if num_clusters < 1:
            with pytest.raises(ValueError):
                clts = DTWKmeans(num_clust = num_clusters)
        else:
            clts = DTWKmeans(num_clust = num_clusters)
            assert clts

    @pytest.mark.parametrize("iterations", [0, -1, 1, 10, 100]) 
    def test_DTWKmeans_init_iterations_acceptance_range(self,iterations):
        num_clusters = 5
        if iterations < 1:
            with pytest.raises(ValueError):
                clts = DTWKmeans(num_clust = num_clusters, num_iter = iterations)
        else:
            clts = DTWKmeans(num_clust = num_clusters, num_iter = iterations ) 
            assert clts

    @pytest.mark.parametrize("initializations", [0, -1, 1, 10, 100]) 
    def test_DTWKmeans_init_num_initializations_acceptance_range(self,initializations):
        num_clusters = 5
        if initializations < 1:
            with pytest.raises(ValueError):
                clts = DTWKmeans(num_clust = num_clusters, num_init= initializations)
        else:
            clts = DTWKmeans(num_clust = num_clusters, num_init= initializations ) 
            assert clts

    @pytest.mark.parametrize("warp", [0, -1, 5, 10]) 
    def test_DTWKmeans_init_warp_acceptance_range(self,warp):
        num_clusters = 5
        if warp < 1:
            with pytest.raises(ValueError):
                clts = DTWKmeans(num_clust = num_clusters, w=warp)
        else:
            clts = DTWKmeans(num_clust = num_clusters, w = warp) 
            assert clts

    @pytest.mark.parametrize("seed", [None, 101]) 
    def test_DTWKmeans_init_random_seed(self,seed):
        num_clusters = 5
        clts = DTWKmeans(num_clust = num_clusters, random_seed = seed) 
        assert clts

    @pytest.mark.parametrize("euclidean,criterion", [(True,'euclidean'), 
                                                     (False, 'cosine')]) 
    def test_DTWKmeans_init_criterion(self,euclidean,criterion):
        num_clusters = 5
        clts = DTWKmeans(num_clust = num_clusters, euclidean=euclidean) 
        assert clts.criterion == criterion


class TestDTWKmeans_features(object):
    def test_DTWKmeans_fit_is_reproduceable_using_random_seed(self):
        list_of_series = make_flat_dataset([-1.0,0,1.0],10,additive_noise_factor=0.1,level_noise_factor=0.1,lengths=[5])
        num_clusters = 3
        iterations = 1
        random_seed = 101
        clts_1 = DTWKmeans(num_clust = num_clusters, num_iter = iterations, random_seed=random_seed)
        clts_1.fit(list_of_series)
        df1 = pd.DataFrame(clts_1.cluster_centers_)
        clts_2 = DTWKmeans(num_clust = num_clusters, num_iter = iterations, random_seed=random_seed)
        clts_2.fit(list_of_series)
        df2 = pd.DataFrame(clts_2.cluster_centers_)
        assert np.all(df1.values==df2.values)

    def test_DTWKmeans_inertia_positive(self):
        list_of_series = make_flat_dataset([-1.0,0,0.5],10,additive_noise_factor=0.3,level_noise_factor=0.3,lengths=[5])
        num_clusters = 3
        iterations = 1
        random_seed = 101
        clts_1 = DTWKmeans(num_clust = num_clusters, num_iter = iterations, random_seed=random_seed)
        clts_1.fit(list_of_series)
        intertia = clts_1._inertia(list_of_series)
        assert intertia > 0

    def test_DTWKmeans_inertia_decrease_with_iteration_increase(self):
        list_of_series = make_flat_dataset([-1.0,0,0.5],10,additive_noise_factor=0.3,level_noise_factor=0.3,lengths=[5])
        num_clusters = 3
        random_seed = 101
        clts_1 = DTWKmeans(num_clust = num_clusters, num_iter=1, random_seed=random_seed)
        clts_1.fit(list_of_series)
        clts_2 = DTWKmeans(num_clust = num_clusters, num_iter=2, random_seed=random_seed)
        clts_2.fit(list_of_series)
        print (clts_1._inertia(list_of_series))
        print (clts_2._inertia(list_of_series))
        #assert False
        assert clts_1._inertia(list_of_series) >= clts_2._inertia(list_of_series)

    @pytest.mark.parametrize("num_init,expected_inertia", [(1,2023.44),(2,664.40)])
    def test_DTWKmeans_single_num_init(self,num_init,expected_inertia):
        list_of_series = flat_dataset(random_seed=101)
        # running only 2 times even if list of random seeds is 10
        random_seed = 22
        clts = DTWKmeans(num_clust = 3, num_iter = 10, num_init = num_init,
                        w=1,euclidean=True,random_seed=random_seed)
        clts.fit(list_of_series)
        inertia=clts._inertia(list_of_series)
        assert inertia == pytest.approx(expected_inertia,abs=1e-2)

def flat_dataset(random_seed=101):
        # build the dataset around 3 levels
        levels = [1.5,0,-1.5]
        # with different number of elements for each cluster
        sizes = [15,30,10]
        # set random seed for reproduceability, you can remove the argument to allow different results for each run
        list_of_series = make_flat_dataset(levels,sizes,  
                                        additive_noise_factor=0.4,level_noise_factor=0.4,
                                        lengths=[10],random_seed=random_seed)
        return list_of_series