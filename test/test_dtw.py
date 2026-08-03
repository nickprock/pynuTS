# unit test suite for the pynuTS DTW engine

import numpy as np
import pytest

from pynuTS.dtw import dtw_distance, dtw_matrix, dtw_path, local_cost_matrix


class TestProperties:
    def test_distance_to_itself_is_zero(self):
        x = np.array([1.0, 3.0, 2.0, 8.0, 4.0])
        assert dtw_distance(x, x) == pytest.approx(0.0)

    def test_symmetry(self):
        x = np.array([1.0, 3.0, 2.0, 8.0, 4.0])
        y = np.array([1.0, 2.0, 2.0, 9.0])
        assert dtw_distance(x, y) == pytest.approx(dtw_distance(y, x))

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.normal(size=int(rng.integers(2, 20)))
            y = rng.normal(size=int(rng.integers(2, 20)))
            assert dtw_distance(x, y) >= 0

    def test_equal_length_distance_is_bounded_by_manhattan(self):
        """with equal lengths the diagonal path is feasible, so DTW cannot cost more"""
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=15), rng.normal(size=15)
        assert dtw_distance(x, y) <= np.abs(x - y).sum() + 1e-9

    def test_narrower_band_never_cheaper(self):
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=25), rng.normal(size=25)
        assert dtw_distance(x, y, w=2) >= dtw_distance(x, y) - 1e-9

    def test_known_value(self):
        """worked out by hand on the example in the docstring"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0])
        # best alignment: 1-1, 2-3, 3-3  ->  0 + 1 + 0 = 1
        assert dtw_distance(x, y) == pytest.approx(1.0)

    def test_constant_shift(self):
        """a ramp against the same ramp shifted by one: the warping path
        absorbs the shift and only the two endpoints cost 1 each"""
        x = np.arange(10.0)
        assert dtw_distance(x, x + 1.0) == pytest.approx(2.0)


class TestMatrix:
    def test_borders_are_infinite(self):
        """a zeroed border would let the path enter the matrix for free and
        silently return a wrong distance"""
        _, DTW = dtw_matrix(np.array([1.0, 2.0]), np.array([5.0, 6.0]))
        assert DTW[0, 0] == 0.0
        assert np.isinf(DTW[0, 1:]).all()
        assert np.isinf(DTW[1:, 0]).all()

    def test_matrix_shape_and_corner(self):
        x, y = np.arange(5.0), np.arange(7.0)
        dist, DTW = dtw_matrix(x, y)
        assert DTW.shape == (6, 8)
        assert DTW[-1, -1] == pytest.approx(dist)

    def test_band_is_widened_to_keep_a_path_feasible(self):
        """with w=1 and very different lengths no path would reach the corner
        unless the band is widened"""
        dist = dtw_distance(np.arange(3.0), np.arange(20.0), w=1)
        assert np.isfinite(dist)


class TestValidation:
    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            dtw_distance(np.array([]), np.array([1.0]))

    def test_bad_criterion_raises(self):
        with pytest.raises(ValueError):
            dtw_distance(np.array([1.0]), np.array([1.0]), criterion='manhattan')

    def test_w_below_one_raises(self):
        with pytest.raises(ValueError):
            dtw_distance(np.arange(4.0), np.arange(4.0), w=0)

    def test_three_dimensional_input_raises(self):
        with pytest.raises(ValueError):
            dtw_distance(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))

    def test_feature_mismatch_raises(self):
        with pytest.raises(ValueError):
            dtw_distance(np.zeros((5, 2)), np.zeros((5, 3)))


class TestMultivariate:
    def test_multivariate_euclidean(self):
        x = np.array([[0.0, 0.0], [3.0, 4.0]])
        y = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert dtw_distance(x, y) == pytest.approx(0.0)

    def test_cost_matrix_shape(self):
        cost = local_cost_matrix(np.arange(4.0), np.arange(6.0))
        assert cost.shape == (4, 6)

    def test_cosine_on_orthogonal_vectors(self):
        x = np.array([[1.0, 0.0]])
        y = np.array([[0.0, 1.0]])
        assert local_cost_matrix(x, y, criterion='cosine')[0, 0] == pytest.approx(1.0)

    def test_cosine_handles_null_vectors(self):
        x = np.array([[0.0, 0.0]])
        y = np.array([[1.0, 1.0]])
        assert np.isfinite(local_cost_matrix(x, y, criterion='cosine')).all()


class TestPath:
    def test_documented_example(self):
        dist, path = dtw_path(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0]))
        assert path == [(0, 0), (1, 0), (2, 1)]
        assert dist == pytest.approx(1.0)

    def test_ties_are_resolved_consistently(self):
        """[(0,0),(1,0),(2,1)] and [(0,0),(1,1),(2,1)] both cost 1.0 here, so
        the test above pins down a choice, not a mathematical truth"""
        x, y = np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0])
        cost = local_cost_matrix(x, y)
        alternative = [(0, 0), (1, 1), (2, 1)]
        dist, _ = dtw_path(x, y)
        assert sum(cost[i, j] for i, j in alternative) == pytest.approx(dist)

    @pytest.mark.parametrize("seed", range(10))
    def test_path_is_a_valid_alignment(self, seed):
        rng = np.random.default_rng(seed)
        n, m = int(rng.integers(2, 25)), int(rng.integers(2, 25))
        x, y = rng.normal(size=n), rng.normal(size=m)

        dist, path = dtw_path(x, y)

        assert path[0] == (0, 0)
        assert path[-1] == (n - 1, m - 1)
        # every element of both series takes part in the alignment
        assert sorted({i for i, _ in path}) == list(range(n))
        assert sorted({j for _, j in path}) == list(range(m))
        # steps are monotone and advance by at most one on each axis
        for (a, b), (c, d) in zip(path, path[1:]):
            assert 0 <= c - a <= 1 and 0 <= d - b <= 1 and (c - a) + (d - b) >= 1

    @pytest.mark.parametrize("seed", range(10))
    def test_path_cost_equals_the_distance(self, seed):
        rng = np.random.default_rng(50 + seed)
        x, y = rng.normal(size=int(rng.integers(2, 20))), rng.normal(size=int(rng.integers(2, 20)))
        dist, path = dtw_path(x, y)
        cost = local_cost_matrix(x, y)
        assert sum(cost[i, j] for i, j in path) == pytest.approx(dist)

    def test_path_stays_inside_the_band(self):
        dist, path = dtw_path(np.arange(20.0), np.arange(20.0) + 0.5, w=3)
        assert all(abs(i - j) <= 3 for i, j in path)

    def test_identical_series_align_on_the_diagonal(self):
        x = np.array([1.0, 4.0, 2.0, 7.0])
        _, path = dtw_path(x, x)
        assert path == [(0, 0), (1, 1), (2, 2), (3, 3)]


class TestSquaredEuclidean:
    def test_is_a_supported_criterion(self):
        assert dtw_distance(np.arange(5.0), np.arange(5.0) + 1, criterion='sqeuclidean') > 0

    def test_zero_on_identical_series(self):
        x = np.array([1.0, 7.0, 3.0])
        assert dtw_distance(x, x, criterion='sqeuclidean') == pytest.approx(0.0)

    def test_squared_distance_is_the_sum_of_squared_local_costs(self):
        """this identity is what makes the DBA descent argument work"""
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=12), rng.normal(size=17)
        dist, path = dtw_path(x, y, criterion='sqeuclidean')
        assert dist ** 2 == pytest.approx(sum((x[i] - y[j]) ** 2 for i, j in path))

    def test_euclidean_is_left_untouched(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=10), rng.normal(size=13)
        dist, path = dtw_path(x, y, criterion='euclidean')
        assert dist == pytest.approx(sum(abs(x[i] - y[j]) for i, j in path))


class TestDeprecatedShim:
    def test_naive_dtw_still_works_and_warns(self):
        from pynuTS.naive_dtw import naive_dtw
        serie_1 = np.array([1, 2, 3, 5, 5, 5, 6], ndmin=2)
        serie_2 = np.array([1, 1, 2, 2, 3, 5], ndmin=2)
        with pytest.warns(DeprecationWarning):
            dist, DTW = naive_dtw(ts1=serie_1, ts2=serie_2, w=1)
        assert np.isfinite(dist)
        assert DTW.shape == (8, 7)
