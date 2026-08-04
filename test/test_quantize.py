# unit test suite for mean-scaled uniform quantization

import numpy as np
import pandas as pd
import pytest

from pynuTS.quantize import MeanScaleQuantizer, mean_scale


class TestMeanScale:
    def test_mean_absolute_value_becomes_one(self):
        rng = np.random.default_rng(0)
        scaled, _ = mean_scale(rng.normal(size=200))
        assert np.mean(np.abs(scaled)) == pytest.approx(1.0)

    def test_sign_is_preserved(self):
        """unlike z-normalization, a positive series stays positive"""
        positive = np.abs(np.random.default_rng(1).normal(size=100)) + 1
        scaled, _ = mean_scale(positive)
        assert (scaled > 0).all()

    def test_scale_is_returned_and_undoes_the_division(self):
        x = np.array([2.0, -4.0, 6.0])
        scaled, scale = mean_scale(x)
        assert scale == pytest.approx(4.0)
        assert np.allclose(scaled * scale, x)

    def test_all_zero_series_does_not_explode(self):
        scaled, scale = mean_scale(np.zeros(10))
        assert np.isfinite(scaled).all() and scale == 1.0

    def test_all_nan_series_does_not_explode(self):
        scaled, scale = mean_scale(np.full(5, np.nan))
        assert scale == 1.0 and scaled.shape == (5,)

    def test_empty_series(self):
        scaled, scale = mean_scale(np.empty(0))
        assert scaled.shape == (0,) and scale == 1.0


class TestConstruction:
    @pytest.mark.parametrize("params,error", [
        ({'n_bins': 1}, ValueError),
        ({'n_bins': 2.5}, TypeError),
        ({'n_bins': True}, TypeError),
        ({'low': 10.0, 'high': 1.0}, ValueError),
        ({'low': 5.0, 'high': 5.0}, ValueError),
        ({'low': np.inf}, ValueError),
        ({'high': np.nan}, ValueError),
    ])
    def test_invalid_parameters(self, params, error):
        with pytest.raises(error):
            MeanScaleQuantizer(**params)

    def test_get_params_round_trip(self):
        params = MeanScaleQuantizer(n_bins=256, low=-8.0, high=8.0).get_params()
        assert params == {'n_bins': 256, 'low': -8.0, 'high': 8.0}

    def test_transform_before_fit_raises(self):
        with pytest.raises(ValueError):
            MeanScaleQuantizer().transform(np.arange(10.0))

    def test_inverse_transform_before_fit_raises(self):
        with pytest.raises(ValueError):
            MeanScaleQuantizer().inverse_transform(np.zeros(5, dtype=int))


class TestGrid:
    def test_fit_does_not_look_at_the_data(self):
        """the same parallel as SAX: a fixed grid is what makes the same token
        mean the same thing for every series"""
        rng = np.random.default_rng(2)
        first = MeanScaleQuantizer(n_bins=64).fit(rng.normal(size=50))
        second = MeanScaleQuantizer(n_bins=64).fit(rng.exponential(size=5000) * 1e6)
        assert np.allclose(first.centers_, second.centers_)

    def test_fit_works_without_any_data(self):
        assert MeanScaleQuantizer(n_bins=32).fit().centers_.shape == (32,)

    def test_centers_span_the_requested_range(self):
        q = MeanScaleQuantizer(n_bins=100, low=-3.0, high=7.0).fit()
        assert q.centers_[0] == pytest.approx(-3.0)
        assert q.centers_[-1] == pytest.approx(7.0)
        assert q.edges_.shape == (99,)

    def test_bin_width(self):
        q = MeanScaleQuantizer(n_bins=11, low=0.0, high=10.0).fit()
        assert q.bin_width() == pytest.approx(1.0)


class TestTokens:
    def test_tokens_stay_in_range(self):
        rng = np.random.default_rng(3)
        q = MeanScaleQuantizer(n_bins=128).fit()
        tokens = q.transform(rng.normal(size=300) * 1e4)
        assert tokens.min() >= 0 and tokens.max() < 128

    def test_tokens_are_scale_invariant(self):
        rng = np.random.default_rng(4)
        q = MeanScaleQuantizer(n_bins=512).fit()
        x = rng.normal(size=80)
        assert np.array_equal(q.transform(x), q.transform(x * 1000))

    def test_tokens_are_not_offset_invariant(self):
        """mean scaling keeps the zero, so a shift is real information"""
        rng = np.random.default_rng(5)
        q = MeanScaleQuantizer(n_bins=512).fit()
        x = rng.normal(size=80)
        assert not np.array_equal(q.transform(x), q.transform(x + 50))

    def test_monotone_values_give_monotone_tokens(self):
        q = MeanScaleQuantizer(n_bins=256).fit()
        tokens = q.transform(np.linspace(-5.0, 5.0, 50))
        assert np.all(np.diff(tokens) >= 0)

    def test_extreme_values_saturate_instead_of_stretching_the_grid(self):
        q = MeanScaleQuantizer(n_bins=256).fit()
        tokens = q.transform(np.concatenate([np.ones(99), [10000.0]]))
        assert tokens[-1] == 255

    def test_accepts_a_pandas_series(self):
        q = MeanScaleQuantizer(n_bins=64).fit()
        values = np.linspace(1.0, 20.0, 30)
        assert np.array_equal(q.transform(pd.Series(values)), q.transform(values))

    def test_three_dimensional_input_raises(self):
        with pytest.raises(TypeError):
            MeanScaleQuantizer().fit().transform(np.zeros((2, 3, 4)))


class TestBatch:
    def test_shape_is_preserved(self):
        rng = np.random.default_rng(6)
        q = MeanScaleQuantizer(n_bins=128).fit()
        assert q.transform(rng.normal(size=(5, 40))).shape == (5, 40)

    def test_each_series_gets_its_own_scale(self):
        q = MeanScaleQuantizer(n_bins=128).fit()
        batch = np.ones((3, 10)) * np.array([[1.0], [10.0], [100.0]])
        q.transform(batch)
        assert np.allclose(q.scales_, [1.0, 10.0, 100.0])

    def test_batch_round_trip(self):
        rng = np.random.default_rng(7)
        q = MeanScaleQuantizer(n_bins=4096).fit()
        batch = rng.normal(size=(4, 60)) * np.array([[1.0], [10.0], [100.0], [0.1]])
        assert np.allclose(q.inverse_transform(q.transform(batch)), batch,
                           atol=np.abs(batch).max() * 0.02)


class TestRoundTrip:
    def test_error_is_bounded_by_half_a_bin(self):
        """the guarantee a lossy encoder owes its user: how wrong can it be"""
        rng = np.random.default_rng(8)
        for _ in range(200):
            n_bins = int(rng.integers(16, 4097))
            q = MeanScaleQuantizer(n_bins=n_bins).fit()
            x = rng.normal(size=int(rng.integers(10, 200))) * rng.uniform(0.01, 1000)

            reconstructed = q.inverse_transform(q.transform(x))
            scale = q.scales_[0]
            scaled = x / scale
            inside = (scaled > q.low) & (scaled < q.high)
            if not inside.any():
                continue
            bound = q.bin_width() / 2 * scale + 1e-9
            assert np.abs(reconstructed - x)[inside].max() <= bound

    def test_more_bins_never_hurt(self):
        x = np.sin(np.linspace(0, 20, 500)) * 37 + 120
        errors = []
        for n_bins in (16, 64, 256, 1024, 4096):
            q = MeanScaleQuantizer(n_bins=n_bins).fit()
            errors.append(np.abs(q.inverse_transform(q.transform(x)) - x).max())
        assert all(a > b for a, b in zip(errors, errors[1:]))

    def test_inverse_transform_accepts_an_explicit_scale(self):
        q = MeanScaleQuantizer(n_bins=256).fit()
        tokens = q.transform(np.linspace(-3.0, 3.0, 20))
        assert np.allclose(q.inverse_transform(tokens, scale=q.scales_[0]),
                           q.inverse_transform(tokens))

    def test_inverse_transform_without_any_scale_raises(self):
        q = MeanScaleQuantizer(n_bins=64).fit()
        with pytest.raises(ValueError, match="scale"):
            q.inverse_transform(np.zeros(5, dtype=int))

    def test_wrong_number_of_scales_raises(self):
        q = MeanScaleQuantizer(n_bins=64).fit()
        with pytest.raises(ValueError, match="scales"):
            q.inverse_transform(np.zeros((3, 5), dtype=int), scale=[1.0, 2.0])

    @pytest.mark.parametrize("bad", [-1, 64])
    def test_out_of_range_tokens_raise(self, bad):
        q = MeanScaleQuantizer(n_bins=64).fit()
        with pytest.raises(ValueError, match="token"):
            q.inverse_transform(np.array([bad]), scale=1.0)
