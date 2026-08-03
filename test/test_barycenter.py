# unit test suite for DTW barycenter averaging

import numpy as np
import pytest

from pynuTS.barycenter import dba, medoid_index
from pynuTS.dtw import dtw_distance


def wgss(center, group, criterion="sqeuclidean", w=None):
    """within-group sum of squared DTW distances, the quantity DBA minimizes"""
    return sum(dtw_distance(center, s, w=w, criterion=criterion) ** 2 for s in group)


def shifted_peaks(shifts=(-9, -4, 0, 4, 9), length=60):
    base = np.exp(-np.linspace(-3, 3, length) ** 2)
    return base, [np.roll(base, k) for k in shifts]


class TestMedoid:
    def test_medoid_of_a_group_with_one_obvious_center(self):
        group = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])]
        assert medoid_index(group) == 1

    def test_medoid_of_identical_series(self):
        x = np.array([1.0, 2.0, 3.0])
        assert medoid_index([x, x, x]) == 0

    def test_empty_group_raises(self):
        with pytest.raises(ValueError):
            medoid_index([])


class TestBasics:
    def test_average_of_identical_series_is_that_series(self):
        x = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        assert np.allclose(dba([x, x, x]), x)

    def test_output_is_one_dimensional_for_univariate_input(self):
        group = [np.arange(10.0), np.arange(10.0) + 1]
        assert dba(group).ndim == 1

    def test_init_fixes_the_output_length(self):
        group = [np.random.default_rng(0).normal(size=n) for n in (8, 15, 22)]
        assert dba(group, init=np.zeros(13)).shape == (13,)

    def test_series_of_different_lengths(self):
        rng = np.random.default_rng(1)
        group = [rng.normal(size=int(n)) for n in (8, 12, 19, 25)]
        result = dba(group)
        assert result.ndim == 1 and np.isfinite(result).all()

    def test_multivariate(self):
        rng = np.random.default_rng(2)
        group = [rng.normal(size=(int(n), 3)) for n in (10, 14, 18)]
        result = dba(group)
        assert result.ndim == 2 and result.shape[1] == 3 and np.isfinite(result).all()

    def test_single_series(self):
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert np.allclose(dba([x]), x)

    def test_empty_group_raises(self):
        with pytest.raises(ValueError):
            dba([])

    def test_max_iter_below_one_raises(self):
        with pytest.raises(ValueError):
            dba([np.arange(5.0)], max_iter=0)

    def test_feature_mismatch_raises(self):
        with pytest.raises(ValueError):
            dba([np.zeros((5, 2)), np.zeros((5, 3))])

    def test_init_feature_mismatch_raises(self):
        with pytest.raises(ValueError):
            dba([np.zeros((5, 2))], init=np.zeros((5, 3)))


class TestWhyDbaExists:
    def test_the_euclidean_mean_flattens_a_shifted_peak_and_dba_does_not(self):
        """the textbook demonstration: averaging shifted copies of the same peak
        along the index smears it into something none of the inputs looked like"""
        base, group = shifted_peaks()

        euclidean_mean = np.mean(group, axis=0)
        barycenter = dba(group, max_iter=20)

        assert euclidean_mean.max() < 0.8 * base.max()
        assert barycenter.max() > 0.95 * base.max()

    def test_dba_beats_the_euclidean_mean_on_the_dtw_objective(self):
        _, group = shifted_peaks()
        assert wgss(dba(group, max_iter=20), group) < wgss(np.mean(group, axis=0), group)


class TestDescentGuarantee:
    @pytest.mark.parametrize("seed", range(8))
    def test_each_pass_never_increases_the_objective(self, seed):
        """DBA is a descent method: this only holds because the warping path is
        optimal for the very quantity the mean update minimizes, which is why
        'sqeuclidean' is the default criterion"""
        rng = np.random.default_rng(seed)
        group = [rng.normal(size=int(rng.integers(10, 30))) for _ in range(int(rng.integers(2, 7)))]
        init = group[medoid_index(group)]

        previous = wgss(init, group)
        for passes in range(1, 7):
            current = wgss(dba(group, init=init, max_iter=passes), group)
            assert current <= previous + 1e-9
            previous = current

    @pytest.mark.parametrize("seed", range(4))
    def test_descent_holds_with_a_sakoe_chiba_band(self, seed):
        rng = np.random.default_rng(100 + seed)
        group = [rng.normal(size=int(rng.integers(15, 30))) for _ in range(5)]
        init = group[medoid_index(group, w=4)]

        previous = wgss(init, group, w=4)
        for passes in range(1, 6):
            current = wgss(dba(group, init=init, max_iter=passes, w=4), group, w=4)
            assert current <= previous + 1e-9
            previous = current

    @pytest.mark.parametrize("seed", range(4))
    def test_descent_holds_on_multivariate_series(self, seed):
        rng = np.random.default_rng(200 + seed)
        group = [rng.normal(size=(int(rng.integers(10, 25)), 3)) for _ in range(4)]
        init = group[medoid_index(group)]

        previous = wgss(init, group)
        for passes in range(1, 6):
            current = wgss(dba(group, init=init, max_iter=passes), group)
            assert current <= previous + 1e-9
            previous = current

    def test_convergence_stops_early(self):
        """on identical series the first pass already lands on the answer"""
        x = np.arange(20.0)
        assert np.allclose(dba([x, x], max_iter=1), dba([x, x], max_iter=50))
