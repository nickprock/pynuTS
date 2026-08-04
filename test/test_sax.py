# unit test suite for canonical SAX

import numpy as np
import pandas as pd
import pytest

from pynuTS.sax import SAX, gaussian_breakpoints, norm_ppf, paa, znorm


class TestNormPpf:
    def test_median_is_zero(self):
        assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("p,expected", [(0.975, 1.959964), (0.95, 1.644854),
                                            (0.84134475, 1.0), (0.025, -1.959964)])
    def test_known_quantiles(self, p, expected):
        assert float(norm_ppf(p)) == pytest.approx(expected, abs=1e-6)

    def test_symmetry(self):
        p = np.array([0.01, 0.1, 0.3, 0.45])
        assert np.allclose(norm_ppf(p), -norm_ppf(1 - p), atol=1e-9)

    def test_monotone(self):
        values = norm_ppf(np.linspace(0.001, 0.999, 500))
        assert np.all(np.diff(values) > 0)

    def test_round_trip_through_the_cdf(self):
        """Phi(norm_ppf(p)) has to come back to p"""
        import math
        p = np.linspace(0.001, 0.999, 200)
        cdf = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in norm_ppf(p)])
        assert np.allclose(cdf, p, atol=1e-12)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_raises(self, bad):
        with pytest.raises(ValueError):
            norm_ppf(bad)


class TestBreakpoints:
    def test_values_from_the_paper(self):
        assert np.allclose(gaussian_breakpoints(3), [-0.4307273, 0.4307273], atol=1e-6)
        assert np.allclose(gaussian_breakpoints(4), [-0.6744898, 0.0, 0.6744898], atol=1e-6)

    def test_count_and_order(self):
        for size in range(2, 21):
            breakpoints = gaussian_breakpoints(size)
            assert breakpoints.shape == (size - 1,)
            assert np.all(np.diff(breakpoints) > 0)

    def test_symmetric_around_zero(self):
        assert np.allclose(gaussian_breakpoints(6), -gaussian_breakpoints(6)[::-1], atol=1e-12)

    def test_alphabet_below_two_raises(self):
        with pytest.raises(ValueError):
            gaussian_breakpoints(1)


class TestZnorm:
    def test_zero_mean_unit_variance(self):
        out = znorm(np.array([1.0, 5.0, 3.0, 9.0, 2.0]))
        assert out.mean() == pytest.approx(0.0, abs=1e-12)
        assert out.std() == pytest.approx(1.0, abs=1e-12)

    def test_constant_series_does_not_explode(self):
        out = znorm(np.full(10, 7.0))
        assert np.all(out == 0.0) and np.isfinite(out).all()

    def test_offset_and_scale_are_removed(self):
        x = np.array([1.0, 4.0, 2.0, 8.0])
        assert np.allclose(znorm(x), znorm(x * 13.0 + 500.0))


class TestPaa:
    def test_means_and_lengths(self):
        means, lengths = paa(np.arange(10.0), 3)
        assert np.allclose(means, [1.0, 4.0, 7.0, 9.0])
        assert list(lengths) == [3, 3, 3, 1]

    def test_empty(self):
        means, lengths = paa(np.empty(0), 3)
        assert means.shape[0] == 0 and lengths.shape[0] == 0

    def test_window_below_one_raises(self):
        with pytest.raises(ValueError):
            paa(np.arange(5.0), 0)


class TestEncoding:
    def test_alphabet_as_int_uses_letters(self):
        sax = SAX(alphabet=3, windows=1).fit()
        assert set(sax.transform(np.arange(30.0))) <= set("abc")

    def test_alphabet_as_explicit_list(self):
        sax = SAX(alphabet=['low', 'mid', 'high'], windows=10).fit()
        assert sax.transform(np.arange(30.0)) == 'lowmidhigh'

    def test_word_length(self):
        sax = SAX(alphabet=4, windows=7).fit()
        assert len(sax.transform(np.arange(30.0))) == 5   # ceil(30 / 7)

    def test_batch_of_series(self):
        sax = SAX(alphabet=3, windows=5).fit()
        encoded = sax.transform(np.random.default_rng(0).normal(size=(4, 20)))
        assert isinstance(encoded, list) and len(encoded) == 4
        assert all(len(word) == 4 for word in encoded)

    def test_accepts_a_pandas_series(self):
        sax = SAX(alphabet=3, windows=5).fit()
        values = np.arange(20.0)
        assert sax.transform(pd.Series(values)) == sax.transform(values)

    def test_empty_series(self):
        assert SAX().fit().transform(np.empty(0)) == ''

    def test_constant_series_maps_to_the_middle_symbol(self):
        assert SAX(alphabet=3, windows=2).fit().transform(np.full(6, 4.0)) == 'bbb'

    def test_transform_before_fit_raises(self):
        with pytest.raises(ValueError):
            SAX().transform(np.arange(10.0))

    def test_fully_missing_window_is_reported(self):
        sax = SAX(alphabet=3, windows=2).fit()
        with pytest.raises(ValueError, match="missing values"):
            sax.transform(np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0]))

    def test_three_dimensional_input_raises(self):
        with pytest.raises(TypeError):
            SAX().fit().transform(np.zeros((2, 3, 4)))


class TestInvariance:
    """what z-normalization buys, stated as a property rather than a side effect"""

    @pytest.mark.parametrize("offset,scale", [(100.0, 1.0), (0.0, 50.0), (-7.5, 0.02)])
    def test_encoding_is_invariant_to_offset_and_scale(self, offset, scale):
        sax = SAX(alphabet=4, windows=25).fit()
        x = np.sin(np.linspace(0, 6, 100))
        assert sax.transform(x) == sax.transform(x * scale + offset)

    def test_breakpoints_do_not_depend_on_the_data(self):
        """the reason encodings are comparable across series"""
        rng = np.random.default_rng(0)
        first = SAX(alphabet=5).fit(rng.normal(size=100))
        second = SAX(alphabet=5).fit(rng.exponential(size=3000) * 1000)
        assert np.allclose(first.breakpoints_, second.breakpoints_)

    def test_fit_works_without_any_data(self):
        assert SAX(alphabet=5).fit().breakpoints_.shape == (4,)


class TestMinDist:
    def test_identical_words_have_zero_distance(self):
        sax = SAX(alphabet=4, windows=5).fit()
        word = sax.transform(np.arange(20.0))
        assert sax.mindist(word, word, n=20) == pytest.approx(0.0)

    def test_adjacent_symbols_contribute_nothing(self):
        """two segments either side of a breakpoint can be arbitrarily close"""
        sax = SAX(alphabet=4, windows=1).fit()
        assert sax.mindist('a', 'b', n=1) == pytest.approx(0.0)
        assert sax.mindist('a', 'c', n=1) > 0.0

    def test_symmetry(self):
        sax = SAX(alphabet=5, windows=4).fit()
        rng = np.random.default_rng(1)
        a, b = sax.transform(rng.normal(size=20)), sax.transform(rng.normal(size=20))
        assert sax.mindist(a, b, n=20) == pytest.approx(sax.mindist(b, a, n=20))

    def test_length_mismatch_raises(self):
        sax = SAX(alphabet=3, windows=2).fit()
        with pytest.raises(ValueError):
            sax.mindist('abc', 'ab', n=6)

    def test_wrong_n_raises(self):
        sax = SAX(alphabet=3, windows=2).fit()
        with pytest.raises(ValueError, match="symbols"):
            sax.mindist('abc', 'abc', n=100)

    def test_unknown_symbol_raises(self):
        sax = SAX(alphabet=3, windows=2).fit()
        with pytest.raises(ValueError, match="alphabet"):
            sax.mindist('abz', 'abc', n=6)

    def test_mindist_before_fit_raises(self):
        with pytest.raises(ValueError):
            SAX().mindist('ab', 'ab', n=8)

    def test_empty_words(self):
        assert SAX().fit().mindist('', '', n=0) == 0.0


class TestLowerBounding:
    """The property the whole technique rests on. If it ever fails, SAX cannot
    be used to prune candidates in a similarity search without losing matches."""

    GENERATORS = {
        'gaussian': lambda rng, n: rng.normal(size=n),
        'random_walk': lambda rng, n: np.cumsum(rng.normal(size=n)),
        'sinusoid': lambda rng, n: np.sin(np.linspace(0, rng.uniform(1, 20), n)),
        'skewed': lambda rng, n: rng.exponential(size=n),
        'discrete': lambda rng, n: np.round(rng.normal(size=n) * 2),
        'extreme_scale': lambda rng, n: rng.normal(size=n) * rng.uniform(0.01, 1000) + rng.uniform(-500, 500),
    }

    @pytest.mark.parametrize("family", sorted(GENERATORS))
    def test_mindist_never_exceeds_the_euclidean_distance(self, family):
        generator = self.GENERATORS[family]
        rng = np.random.default_rng(hash(family) % 2 ** 32)

        for _ in range(300):
            alphabet = int(rng.integers(2, 11))
            windows = int(rng.integers(1, 12))
            n = int(rng.integers(8, 120))
            sax = SAX(alphabet=alphabet, windows=windows).fit()

            x, y = generator(rng, n), generator(rng, n)
            lower_bound = sax.mindist(sax.transform(x), sax.transform(y), n=n)
            true_distance = np.linalg.norm(znorm(x) - znorm(y))

            assert lower_bound <= true_distance + 1e-9, (
                "lower bound violated with alphabet=%d windows=%d n=%d: %f > %f"
                % (alphabet, windows, n, lower_bound, true_distance))

    def test_bound_holds_when_the_last_window_is_short(self):
        """n not divisible by the window size is the case where a formula that
        assumes equal segments would quietly break the guarantee"""
        rng = np.random.default_rng(42)
        sax = SAX(alphabet=5, windows=7).fit()
        for _ in range(200):
            n = int(rng.integers(8, 60))
            if n % 7 == 0:
                continue
            x, y = rng.normal(size=n), rng.normal(size=n)
            assert sax.mindist(sax.transform(x), sax.transform(y), n=n) <= \
                   np.linalg.norm(znorm(x) - znorm(y)) + 1e-9

    def test_bound_is_not_trivially_zero(self):
        """a bound that always returned 0 would pass the test above and be useless"""
        rng = np.random.default_rng(3)
        sax = SAX(alphabet=8, windows=4).fit()
        positive = 0
        for _ in range(100):
            x, y = np.cumsum(rng.normal(size=40)), np.cumsum(rng.normal(size=40))
            if sax.mindist(sax.transform(x), sax.transform(y), n=40) > 0:
                positive += 1
        assert positive > 60


class TestConstruction:
    @pytest.mark.parametrize("params,error", [
        ({'alphabet': 1}, ValueError),
        ({'alphabet': 27}, ValueError),
        ({'alphabet': ['a']}, ValueError),
        ({'alphabet': ['a', 'a', 'b']}, ValueError),
        ({'alphabet': 'abc'}, TypeError),
        ({'alphabet': True}, TypeError),
        ({'windows': 0}, ValueError),
        ({'windows': 2.5}, TypeError),
        ({'znormalize': 'yes'}, TypeError),
    ])
    def test_invalid_parameters(self, params, error):
        with pytest.raises(error):
            SAX(**params)

    def test_get_params_round_trip(self):
        params = SAX(alphabet=6, windows=3, znormalize=False).get_params()
        assert params == {'alphabet': 6, 'windows': 3, 'znormalize': False}

    def test_znormalize_off_keeps_the_raw_level(self):
        sax = SAX(alphabet=3, windows=2, znormalize=False).fit()
        assert sax.transform(np.full(4, -5.0)) != sax.transform(np.full(4, 5.0))
