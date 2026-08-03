"""
Created on Wed Apr 01 2020

@project: pynuTS
@author: nicola procopio
@last_update: 03/08/2026
@decription: impute missing value with rolling mean
@reference: https://iaml.it/blog/serie-storiche-1-dati-mancanti
"""
import numpy as np
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class TsImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in the time series with rolling mean

    Parameters
    -----------------------
    m_avg : int
        default 1. The range of the moving average.
    copy : bool
        default True. If true create a copy of X the input, else overwrite.

    Returns
    ----------------------
    a 1D numpy array, or a pandas Series if a pandas Series was passed in

    Examples
    ----------------------
    >> import numpy as np
    >> from pynuTS.impute import TsImputer, maximum_distance_recommended
    >> X = np.array([1, 2, np.nan, 3, 5, np.nan])
    >> dist = maximum_distance_recommended(X)
    >> imputer = TsImputer(m_avg = dist)
    >> X_new = imputer.fit_transform(X)
    """
    def __init__(self, m_avg: int = 1, copy : bool = True):
        if (m_avg is None) or isinstance(m_avg, bool) or not isinstance(m_avg, (int, np.integer)):
            raise ValueError ("m_avg must be a positive integer")
        if m_avg < 1:
            raise ValueError ("m_avg must be a positive integer")

        self.m_avg = m_avg
        self.copy = copy

    def fit(self, X, y=None):
        """This imputer is stateless: fit only validates the input."""
        _to_values(X)
        return self

    def transform(self, X):
        """
        Replace every missing value with the mean of the surrounding window.

        Parameters
        -----------------------
        X : 1D numpy array or pandas Series

        Returns
        -----------------------
        the imputed series, same type and index as the input
        """
        # work positionally on a float array: indexing a pandas Series by
        # position is not the same as indexing it by label, and on a
        # DatetimeIndex the old code appended spurious rows instead of imputing
        source = _to_values(X)
        # target is always a distinct array so that the loop keeps reading the
        # pristine series
        target = np.array(source, copy=True)

        na_list = np.where(np.isnan(source))[0].tolist()
        for i in tqdm(na_list):
            # every imputed value is computed from the *original* series, so
            # long runs of missing values do not drift towards a constant
            target[i] = _moving_average(source, i, self.m_avg)

        if isinstance(X, Series):
            return Series(target, index=X.index, name=X.name)
        if not self.copy and isinstance(X, np.ndarray) and X.dtype.kind == 'f':
            # only a mutable float array can take the result back in place
            X[...] = target
            return X
        return target

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)


def _to_values(X):
    if isinstance(X, Series):
        values = X.values
    elif isinstance(X, (list, tuple, np.ndarray)):
        values = np.asarray(X)
    else:
        raise TypeError("X must be a numpy.array or a list or a pandas Series")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise TypeError("X must be a 1-D array")
    return values


def _moving_average(data, position, m_avg):
    low_bound = max(position - m_avg, 0)
    upp_bound = min(position + m_avg + 1, len(data))
    window = data[low_bound:upp_bound]
    known = window[~np.isnan(window)]
    if known.size == 0:
        raise ValueError(
            "the window around position %d contains no known value, "
            "increase m_avg" % position
        )
    return known.mean()


def maximum_distance_recommended(X):
    """
    Recommend the maximum range without missing value

    Parameters
    -----------------------
    X : 1D numpy array

    Returns
    -----------------------
    max_range : int.
        the maximum distance recommended

    Examples
    ----------------------
    >> import numpy as np
    >> from pynuTS.impute import maximum_distance_recommended
    >> X = np.array([1,2,np.nan,3, 5, np.nan])
    >> dist = maximum_distance_recommended(X)
        the maximum range recommended for the 'm_avg' parameter is 1
    """
    values = _to_values(X)
    na_list = np.where(np.isnan(values))[0].tolist()
    gaps = sorted(j - i for i, j in zip(na_list[:-1], na_list[1:]))

    if len(gaps) > 0:
        # gaps of 1 mean consecutive missing values: no window fits between
        # them, so look for the smallest gap that actually leaves room
        usable = [g for g in gaps if g > 1]
        max_range = usable[0] - 1 if usable else 1
    else:
        max_range = max(int((values.shape[0] - 1) / 2) - 1, 1)
    print("the maximum range recommended for the 'm_avg' parameter is {0} ".format(max_range))
    return max_range
