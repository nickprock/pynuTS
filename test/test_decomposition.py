# unit test suite for pynuTS decomposition

import numpy as np
import pandas as pd
import pytest

from pynuTS.decomposition import NaiveSAX


class TestBasicObject:
    def test_default_object_contruction(self):
        sax = NaiveSAX()
        assert sax

    @pytest.mark.parametrize("init_params,expected_exception",
                [({'levels': 1}, TypeError),
                 ({'bounds': 1}, TypeError),
                 ({'windows': [0.5, 0.6]}, TypeError),
                 ({'quantile': [0.5, 0.6]}, TypeError),
                 # a levels/bounds length mismatch is a value problem, not a type one
                 ({'levels': ['A', 'B'], 'bounds': [0.25, 0.75]}, ValueError),
                 ({'windows': 0}, ValueError),
                 # bounds must be usable as quantiles
                 ({'bounds': [0.75, 0.25]}, ValueError),
                 ({'bounds': [0.25, 1.75]}, ValueError),
                 ])
    def test_bad_parameters(self, init_params, expected_exception):
        with pytest.raises(expected_exception):
            NaiveSAX(**init_params)

    def test_absolute_bounds_may_fall_outside_zero_one(self):
        assert NaiveSAX(quantile=False, bounds=[3, 6])


class TestFitTransform:
    @pytest.mark.parametrize("input_series,expected_encoding", [([], '')])
    def test_corner_cases(self, input_series, expected_encoding):
        sax = NaiveSAX()
        assert sax.fit_transform(input_series) == expected_encoding

    @pytest.mark.parametrize("input_series_len,window,expected_encoding_len",
                        [(10, 1, 10), (10, 2, 5), (10, 3, 4), (10, 5, 2), (10, 8, 2), (10, 12, 1)])
    def test_encoded_len(self, input_series_len, window, expected_encoding_len):
        X = np.zeros(input_series_len)
        sax = NaiveSAX(windows=window)
        assert len(sax.fit_transform(X)) == expected_encoding_len

    @pytest.mark.parametrize("window,expected_encoding",
                        [(1, 'AAABBBBCCC'), (2, 'ABBCC'), (3, 'ABBC'), (5, 'AC'), (8, 'AC'), (12, 'C')])
    def test_encoding_quantile(self, window, expected_encoding):
        X = np.arange(0.0, 10.0)
        sax = NaiveSAX(windows=window, bounds=[0.25, 0.75], levels=['A', 'B', 'C'])
        assert sax.fit_transform(X) == expected_encoding

    @pytest.mark.parametrize("window,expected_encoding",
                        [(1, 'AAABBBCCCC'), (2, 'AABCC'), (3, 'ABCC'), (5, 'AC'), (8, 'BC'), (12, 'B')])
    def test_encoding_absolute(self, window, expected_encoding):
        X = np.arange(0.0, 10.0)
        sax = NaiveSAX(windows=window, quantile=False, bounds=[3, 6], levels=['A', 'B', 'C'])
        assert sax.fit_transform(X) == expected_encoding

    @pytest.mark.parametrize("container", [list, np.asarray, pd.Series])
    def test_accepts_list_array_and_series(self, container):
        X = container(np.arange(0.0, 10.0).tolist() if container is list else np.arange(0.0, 10.0))
        assert NaiveSAX(windows=2).fit_transform(X) == NaiveSAX(windows=2).fit_transform(np.arange(0.0, 10.0))

    def test_two_dimensional_input_raises(self):
        with pytest.raises(TypeError):
            NaiveSAX().fit_transform(np.zeros((3, 4)))

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            NaiveSAX().fit_transform("not a series")


class TestMissingValues:
    def test_a_single_nan_does_not_wipe_out_the_whole_encoding(self):
        """regression: NaN used to poison every quantile, so the encoder
        silently returned an empty string for the entire series"""
        X = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        encoded = NaiveSAX(windows=2).fit_transform(X)
        assert len(encoded) == 3

    def test_a_fully_missing_window_is_reported(self):
        X = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
        with pytest.raises(ValueError, match="missing values"):
            NaiveSAX(windows=2).fit_transform(X)


class TestSklearnApi:
    def test_fit_then_transform(self):
        sax = NaiveSAX(windows=2)
        assert sax.fit(np.arange(0.0, 10.0)) is sax
        assert sax.transform(np.arange(0.0, 10.0)) == 'ABBCC'

    def test_transform_before_fit_raises(self):
        with pytest.raises(ValueError):
            NaiveSAX().transform(np.arange(10.0))

    def test_breakpoints_learned_once_make_series_comparable(self):
        """fitting per series makes the encoding scale invariant, which is the
        reason SAX strings could not be compared across series"""
        low = np.arange(0.0, 10.0)
        high = low + 1000.0

        per_series = (NaiveSAX(windows=2).fit_transform(low),
                      NaiveSAX(windows=2).fit_transform(high))
        assert per_series[0] == per_series[1]  # indistinguishable, the bug

        shared = NaiveSAX(windows=2).fit(np.concatenate([low, high]))
        assert shared.transform(low) != shared.transform(high)

    def test_get_params_round_trip(self):
        sax = NaiveSAX(levels=['x', 'y'], bounds=[0.5], windows=4, quantile=False)
        params = sax.get_params()
        assert params['levels'] == ['x', 'y']
        assert params['bounds'] == [0.5]
        assert params['windows'] == 4
        assert params['quantile'] is False
