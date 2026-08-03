# unit test suite for pynuTS clustering

import numpy as np
import pandas as pd
import pytest

from pynuTS.clustering import DTWKmeans
from pynuTS.datasets import make_flat_dataset, make_slopes_dataset


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
        clts = DTWKmeans(num_clust=num_clusters, num_iter=iterations)
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
        slopes = [0.3, 0, -0.3]
        list_of_series = make_slopes_dataset(slopes, 10, additive_noise_factor=0.0,
                                             intercept_noise_factor=0.0, lengths=[3])
        # setting random seed othewise fitting may end with two matching centroids
        clts = DTWKmeans(num_clust=3, num_iter=10, w=1, criterion='euclidean', seed=101)

        clts.fit(list_of_series)

        df_data = pd.DataFrame(list_of_series).drop_duplicates().sort_values(by=2)
        df_centroids = pd.DataFrame(clts.cluster_centers_).drop_duplicates().sort_values(by=2)
        assert np.allclose(df_data, df_centroids)


class TestDTWKmeans_init(object):
    def test_DTWKmeans_init_default_kwargs(self):
        num_clusters = 2
        clts = DTWKmeans(num_clust=num_clusters)
        assert clts

    @pytest.mark.parametrize("num_clusters", [0, -1, 1, 5, 100])
    def test_DTWKmeans_init_clusters_acceptance_range(self, num_clusters):
        if num_clusters < 1:
            with pytest.raises(ValueError):
                DTWKmeans(num_clust=num_clusters)
        else:
            assert DTWKmeans(num_clust=num_clusters)

    @pytest.mark.parametrize("iterations", [0, -1, 1, 10, 100])
    def test_DTWKmeans_init_iterations_acceptance_range(self, iterations):
        if iterations < 1:
            with pytest.raises(ValueError):
                DTWKmeans(num_clust=5, num_iter=iterations)
        else:
            assert DTWKmeans(num_clust=5, num_iter=iterations)

    @pytest.mark.parametrize("initializations", [0, -1, 1, 10, 100])
    def test_DTWKmeans_init_num_initializations_acceptance_range(self, initializations):
        if initializations < 1:
            with pytest.raises(ValueError):
                DTWKmeans(num_clust=5, num_init=initializations)
        else:
            assert DTWKmeans(num_clust=5, num_init=initializations)

    @pytest.mark.parametrize("warp", [0, -1, 5, 10])
    def test_DTWKmeans_init_warp_acceptance_range(self, warp):
        if warp < 1:
            with pytest.raises(ValueError):
                DTWKmeans(num_clust=5, w=warp)
        else:
            assert DTWKmeans(num_clust=5, w=warp)

    @pytest.mark.parametrize("seed", [None, 101])
    def test_DTWKmeans_init_random_seed(self, seed):
        assert DTWKmeans(num_clust=5, seed=seed)

    @pytest.mark.parametrize("criterion", ['sqeuclidean', 'euclidean', 'cosine'])
    def test_DTWKmeans_init_criterion(self, criterion):
        clts = DTWKmeans(num_clust=5, criterion=criterion)
        assert clts.criterion == criterion

    def test_DTWKmeans_init_rejects_unknown_criterion(self):
        with pytest.raises(ValueError):
            DTWKmeans(num_clust=5, criterion='manhattan')

    @pytest.mark.parametrize("averaging", ['dba', 'mean'])
    def test_DTWKmeans_init_averaging(self, averaging):
        assert DTWKmeans(num_clust=5, averaging=averaging).averaging == averaging

    def test_DTWKmeans_init_rejects_unknown_averaging(self):
        with pytest.raises(ValueError):
            DTWKmeans(num_clust=5, averaging='median')

    def test_DTWKmeans_init_rejects_bad_dba_iter(self):
        with pytest.raises(ValueError):
            DTWKmeans(num_clust=5, dba_iter=0)

    def test_DTWKmeans_defaults_to_dba(self):
        clts = DTWKmeans(num_clust=2)
        assert clts.averaging == 'dba'
        assert clts.criterion == 'sqeuclidean'

    def test_DTWKmeans_is_a_sklearn_estimator(self):
        """get_params/set_params must round-trip, as sklearn's clone relies on them"""
        clts = DTWKmeans(num_clust=3, num_iter=7, w=2, criterion='cosine', seed=42)
        params = clts.get_params()
        assert params['num_clust'] == 3
        assert params['num_iter'] == 7
        assert params['w'] == 2
        assert params['criterion'] == 'cosine'
        assert params['seed'] == 42


class TestDTWKmeans_features(object):
    def test_DTWKmeans_fit_is_reproduceable_using_random_seed(self):
        list_of_series = make_flat_dataset([-1.0, 0, 1.0], 10, additive_noise_factor=0.1,
                                           level_noise_factor=0.1, lengths=[5])
        num_clusters = 3
        iterations = 1
        seed = 101
        clts_1 = DTWKmeans(num_clust=num_clusters, num_iter=iterations, seed=seed)
        clts_1.fit(list_of_series)
        df1 = pd.DataFrame(clts_1.cluster_centers_)
        clts_2 = DTWKmeans(num_clust=num_clusters, num_iter=iterations, seed=seed)
        clts_2.fit(list_of_series)
        df2 = pd.DataFrame(clts_2.cluster_centers_)
        assert np.all(df1.values == df2.values)

    def test_DTWKmeans_does_not_touch_the_global_random_state(self):
        """seeding the estimator must not silently reseed the caller's program"""
        import random
        list_of_series = make_flat_dataset([-1.0, 1.0], 5, lengths=[5], random_seed=7)
        random.seed(12345)
        expected = [random.random() for _ in range(3)]
        random.seed(12345)
        DTWKmeans(num_clust=2, num_iter=2, seed=999).fit(list_of_series)
        assert [random.random() for _ in range(3)] == expected

    def test_DTWKmeans_inertia_positive(self):
        list_of_series = make_flat_dataset([-1.0, 0, 0.5], 10, additive_noise_factor=0.3,
                                           level_noise_factor=0.3, lengths=[5])
        clts_1 = DTWKmeans(num_clust=3, num_iter=1, seed=101)
        clts_1.fit(list_of_series)
        assert clts_1._inertia(list_of_series) > 0
        assert clts_1.inertia_ > 0

    def test_DTWKmeans_inertia_decrease_with_iteration_increase(self):
        list_of_series = make_flat_dataset([-1.0, 0, 0.5], 10, additive_noise_factor=0.3,
                                           level_noise_factor=0.3, lengths=[5])
        clts_1 = DTWKmeans(num_clust=3, num_iter=1, seed=101)
        clts_1.fit(list_of_series)
        clts_2 = DTWKmeans(num_clust=3, num_iter=2, seed=101)
        clts_2.fit(list_of_series)
        assert clts_1._inertia(list_of_series) >= clts_2._inertia(list_of_series)

    def test_DTWKmeans_more_inits_never_worsen_inertia(self):
        """num_init keeps the best run, so inertia can only improve"""
        list_of_series = flat_dataset(random_seed=101)
        single = DTWKmeans(num_clust=3, num_iter=10, num_init=1, w=1, seed=22).fit(list_of_series)
        multiple = DTWKmeans(num_clust=3, num_iter=10, num_init=3, w=1, seed=22).fit(list_of_series)
        assert multiple.inertia_ <= single.inertia_

    def test_DTWKmeans_every_series_is_assigned_exactly_once(self):
        list_of_series = flat_dataset(random_seed=101)
        clts = DTWKmeans(num_clust=3, num_iter=5, seed=11).fit(list_of_series)
        assigned = sorted(i for members in clts.labels_.values() for i in members)
        assert assigned == list(range(len(list_of_series)))

    def test_DTWKmeans_handles_series_of_different_lengths(self):
        """DTW exists to compare series of different lengths: the centroid
        update must not turn them into NaN"""
        list_of_series = make_flat_dataset([-1.0, 1.0], 6, lengths=[5, 8, 11], random_seed=3)
        clts = DTWKmeans(num_clust=2, num_iter=3, seed=5).fit(list_of_series)
        for centroid in clts.cluster_centers_:
            assert not np.isnan(np.asarray(centroid)).any()

    def test_DTWKmeans_accepts_plain_numpy_arrays(self):
        list_of_arrays = [np.asarray(s) for s in make_flat_dataset([-1.0, 1.0], 5, lengths=[6], random_seed=1)]
        clts = DTWKmeans(num_clust=2, num_iter=3, seed=5).fit(list_of_arrays)
        assert clts._inertia(list_of_arrays) >= 0

    def test_DTWKmeans_predict_before_fit_raises(self):
        with pytest.raises(ValueError):
            DTWKmeans(num_clust=2).predict([pd.Series([1.0, 2.0])])

    def test_DTWKmeans_more_clusters_than_series_raises(self):
        with pytest.raises(ValueError):
            DTWKmeans(num_clust=5).fit([pd.Series([1.0, 2.0])])


class TestDTWKmeans_averaging:
    def test_dba_and_mean_both_run_end_to_end(self):
        data = flat_dataset(random_seed=101)
        for averaging in ('dba', 'mean'):
            clts = DTWKmeans(num_clust=3, num_iter=5, averaging=averaging, seed=7).fit(data)
            assert len(clts.cluster_centers_) == 3
            assert clts.inertia_ >= 0

    def test_dba_recovers_a_shape_the_mean_destroys(self):
        """two clusters of shifted peaks: the element-wise mean flattens the
        peak into something no member ever looked like, DBA keeps it"""
        base = np.exp(-np.linspace(-3, 3, 60) ** 2)
        group_a = [pd.Series(np.roll(base, k)) for k in (-9, -5, 0, 5, 9)]
        group_b = [pd.Series(np.roll(base, k) + 5.0) for k in (-9, -5, 0, 5, 9)]
        data = group_a + group_b

        with_dba = DTWKmeans(num_clust=2, num_iter=5, averaging='dba', seed=3).fit(data)
        with_mean = DTWKmeans(num_clust=2, num_iter=5, averaging='mean', seed=3).fit(data)

        peak_height = lambda c: np.asarray(c).max() - np.asarray(c).min()
        assert min(peak_height(c) for c in with_dba.cluster_centers_) > \
               max(peak_height(c) for c in with_mean.cluster_centers_)

    def test_dba_reaches_a_lower_inertia_on_warped_data(self):
        base = np.exp(-np.linspace(-3, 3, 60) ** 2)
        data = [pd.Series(np.roll(base, k)) for k in (-9, -5, 0, 5, 9)]

        with_dba = DTWKmeans(num_clust=1, num_iter=5, averaging='dba', seed=3).fit(data)
        with_mean = DTWKmeans(num_clust=1, num_iter=5, averaging='mean', seed=3).fit(data)

        assert with_dba.inertia_ < with_mean.inertia_

    def test_dba_centroid_keeps_the_length_of_its_seed(self):
        data = make_flat_dataset([-1.0, 1.0], 6, lengths=[7, 13, 20], random_seed=4)
        clts = DTWKmeans(num_clust=2, num_iter=3, seed=9).fit(data)
        lengths = {len(np.asarray(c)) for c in clts.cluster_centers_}
        assert lengths.issubset({7, 13, 20})

    def test_dba_handles_series_of_different_lengths(self):
        data = make_flat_dataset([-1.0, 1.0], 6, lengths=[5, 8, 11], random_seed=3)
        clts = DTWKmeans(num_clust=2, num_iter=3, seed=5).fit(data)
        for centroid in clts.cluster_centers_:
            assert not np.isnan(np.asarray(centroid)).any()

    def test_dba_is_reproduceable(self):
        data = flat_dataset(random_seed=101)
        a = DTWKmeans(num_clust=3, num_iter=4, seed=31).fit(data)
        b = DTWKmeans(num_clust=3, num_iter=4, seed=31).fit(data)
        assert a.inertia_ == pytest.approx(b.inertia_)
        for ca, cb in zip(a.cluster_centers_, b.cluster_centers_):
            assert np.allclose(ca, cb)


def flat_dataset(random_seed=101):
    # build the dataset around 3 levels
    levels = [1.5, 0, -1.5]
    # with different number of elements for each cluster
    sizes = [15, 30, 10]
    # set random seed for reproduceability, you can remove the argument to allow different results for each run
    list_of_series = make_flat_dataset(levels, sizes,
                                       additive_noise_factor=0.4, level_noise_factor=0.4,
                                       lengths=[10], random_seed=random_seed)
    return list_of_series
