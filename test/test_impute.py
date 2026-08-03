# unit test suite for pynuTS imputation

import numpy as np
import pandas as pd
import pytest

from pynuTS.impute import TsImputer, maximum_distance_recommended


class TestConstruction:
    @pytest.mark.parametrize("m_avg", [0, -1, None, 1.5])
    def test_invalid_m_avg_raises(self, m_avg):
        with pytest.raises(ValueError):
            TsImputer(m_avg=m_avg)

    def test_valid_m_avg(self):
        assert TsImputer(m_avg=3).m_avg == 3

    def test_get_params_round_trip(self):
        params = TsImputer(m_avg=4, copy=False).get_params()
        assert params['m_avg'] == 4
        assert params['copy'] is False


class TestNumpyInput:
    def test_docstring_example(self):
        X = np.array([1, 2, np.nan, 3, 5, np.nan])
        out = TsImputer(m_avg=1).fit_transform(X)
        assert not np.isnan(out).any()
        # mean of the neighbours, 2 and 3
        assert out[2] == pytest.approx(2.5)
        # only one neighbour available at the right edge
        assert out[5] == pytest.approx(5.0)

    def test_known_values_are_untouched(self):
        X = np.array([1.0, 2.0, np.nan, 4.0])
        out = TsImputer(m_avg=1).fit_transform(X)
        assert out[0] == 1.0 and out[1] == 2.0 and out[3] == 4.0

    def test_copy_true_leaves_the_input_alone(self):
        X = np.array([1.0, np.nan, 3.0])
        TsImputer(m_avg=1, copy=True).fit_transform(X)
        assert np.isnan(X[1])

    def test_copy_false_writes_in_place(self):
        X = np.array([1.0, np.nan, 3.0])
        out = TsImputer(m_avg=1, copy=False).fit_transform(X)
        assert not np.isnan(X[1])
        assert out[1] == pytest.approx(2.0)

    def test_no_missing_values_is_a_no_op(self):
        X = np.array([1.0, 2.0, 3.0])
        assert np.allclose(TsImputer(m_avg=1).fit_transform(X), X)

    def test_accepts_a_plain_list(self):
        out = TsImputer(m_avg=1).fit_transform([1.0, np.nan, 3.0])
        assert out[1] == pytest.approx(2.0)

    @pytest.mark.parametrize("container", [list, tuple])
    def test_copy_false_on_an_immutable_container_returns_an_array(self, container):
        """nothing can be written back in place, so just hand back the result"""
        out = TsImputer(m_avg=1, copy=False).fit_transform(container([1.0, np.nan, 3.0]))
        assert isinstance(out, np.ndarray)
        assert out[1] == pytest.approx(2.0)

    def test_integer_input_is_not_truncated(self):
        """an int array used to truncate the imputed float"""
        X = np.array([1, np.nan, 2], dtype=float)
        assert TsImputer(m_avg=1).fit_transform(X)[1] == pytest.approx(1.5)


class TestPandasInput:
    def test_series_with_range_index(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0])
        out = TsImputer(m_avg=1).fit_transform(s)
        assert isinstance(out, pd.Series)
        assert len(out) == 4
        assert out.iloc[2] == pytest.approx(3.0)

    def test_series_with_datetime_index(self):
        """regression: positional writes on a labelled Series appended spurious
        rows instead of imputing, and did it silently"""
        index = pd.date_range('2020-01-01', periods=4)
        s = pd.Series([1.0, 2.0, np.nan, 4.0], index=index)
        out = TsImputer(m_avg=1).fit_transform(s)

        assert len(out) == 4
        assert out.index.equals(index)
        assert not out.isna().any()
        assert out.iloc[2] == pytest.approx(3.0)

    def test_series_name_and_index_are_preserved(self):
        index = pd.date_range('2020-01-01', periods=3, freq='h')
        s = pd.Series([1.0, np.nan, 3.0], index=index, name='sensor')
        out = TsImputer(m_avg=1).fit_transform(s)
        assert out.name == 'sensor'
        assert out.index.equals(index)


class TestPropagation:
    def test_imputed_values_are_computed_from_the_original_series(self):
        """regression: each imputed value used to feed the next one, so long
        runs of missing values drifted towards a constant"""
        X = np.array([0.0, np.nan, np.nan, 30.0])
        out = TsImputer(m_avg=2).fit_transform(X)
        # both gaps see the same known neighbours {0, 30}, so both get 15
        assert out[1] == pytest.approx(15.0)
        assert out[2] == pytest.approx(15.0)

    def test_window_without_any_known_value_is_reported(self):
        X = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        with pytest.raises(ValueError, match="no known value"):
            TsImputer(m_avg=1).fit_transform(X)


class TestMaximumDistanceRecommended:
    def test_isolated_missing_values(self):
        X = np.array([1, 2, np.nan, 3, 5, np.nan])
        assert maximum_distance_recommended(X) == 2

    def test_consecutive_missing_values_do_not_crash(self):
        """regression: a gap of 1 made the helper raise IndexError, which is
        exactly the most common real world case"""
        X = np.array([1.0, np.nan, np.nan, 4.0])
        assert maximum_distance_recommended(X) >= 1

    def test_all_gaps_are_consecutive(self):
        X = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        assert maximum_distance_recommended(X) >= 1

    def test_no_missing_values(self):
        X = np.arange(11.0)
        assert maximum_distance_recommended(X) == 4

    def test_single_missing_value(self):
        X = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        assert maximum_distance_recommended(X) >= 1

    def test_result_is_usable_as_m_avg(self):
        """the whole point of the helper is to feed TsImputer"""
        for X in (np.array([1, 2, np.nan, 3, 5, np.nan]),
                  np.array([1.0, np.nan, np.nan, 4.0]),
                  np.arange(11.0)):
            dist = maximum_distance_recommended(X)
            assert TsImputer(m_avg=dist).fit_transform(X) is not None
