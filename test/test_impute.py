import numpy as np
import pytest
from pynuTS.impute import TsImputer, maximum_distance_recommended, impute_with_strategy, analyze_missing_patterns

def test_imputer_basic():
    X = np.array([1.0, np.nan, 3.0])
    imputer = TsImputer(m_avg=1)
    result = imputer.fit_transform(X)
    assert not np.isnan(result).any()

def test_imputer_no_nan():
    X = np.array([1.0, 2.0, 3.0])
    imputer = TsImputer()
    result = imputer.transform(X)
    assert np.allclose(result, X)

def test_imputer_all_nan():
    X = np.array([np.nan, np.nan])
    imputer = TsImputer()
    result = imputer.fit_transform(X)
    assert np.all(result == 0.0)

def test_maximum_distance():
    X = np.array([1, 2, np.nan, 3, np.nan, 4])
    dist = maximum_distance_recommended(X)
    assert isinstance(dist, int)
    assert dist >= 1

def test_impute_with_strategy():
    X = np.array([1.0, np.nan, 2.0])
    result = impute_with_strategy(X, strategy='rolling_mean', m_avg=1)
    assert not np.isnan(result).any()

def test_impute_with_strategy_invalid():
    X = np.array([1.0, np.nan])
    with pytest.raises(ValueError):
        impute_with_strategy(X, strategy='unknown')

def test_analyze_missing_patterns():
    X = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    result = analyze_missing_patterns(X)
    assert result['total_missing'] == 2
    assert result['missing_percentage'] > 0
