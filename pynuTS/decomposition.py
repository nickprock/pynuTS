"""
Created on Wed May 15 2021

@project: pynuTS
@author: nicola procopio
@last_update: 03/08/2026
@description: dimensionality reduction by SAX encoding
@reference: https://iaml.it/blog/serie-storiche-2-sax-encoding
"""

import numpy as np
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin


class NaiveSAX(BaseEstimator, TransformerMixin):
    def __init__(self, levels: list = ["A", "B", "C"], bounds: list = [0.25, 0.75], windows: int = 2, quantile: bool = True):
        """
        SAX Encoding (Symbolic Aggregate approXimation) is the first symbolic representation for time series that allows for dimensionality reduction and indexing with a lower-bounding distance measure.
        SAX was invented by Eamonn Keogh and Jessica Lin in 2002.

        This is a *naive* variant: breakpoints come from the empirical quantiles
        of the data instead of the equiprobable gaussian breakpoints of the
        original paper, and the series is not z-normalized. It therefore does
        **not** provide the MINDIST lower-bounding guarantee, and encodings are
        comparable across series only if you ``fit`` once and ``transform`` many
        (see below).

        Parameters
        -----------------------
        bounds : list
            default [0.25, 0.75]. Limits in which to fit the values of the series. With default values we have 3 levels. [(min, 1st quartile);(1st quartile, 3rd quartile), (3rd quartile, max)].
        levels: list
            default ["A", "B", "C"]. Labels for SAX Encoding.
        windows : int
            default 2. Time window for PAA (Piecewise Aggregate Approximation).
        quantile: bool
            default True. If False the values in bounds are used without apply any function.

        Attributes
        -----------------------
        breakpoints_ : 1D numpy array
            the absolute thresholds separating the levels, available after fit.

        Returns
        -----------------------
        sax_string

        Example
        -----------------------
        >> import numpy as np
        >> ts1 = 2.5 * np.random.randn(100,) + 3
        >> ts2 = 4.5 * np.random.randn(100,) + 13
        >> from pynuTS.decomposition import NaiveSAX
        >> sax = NaiveSAX()
        >> ts1_decomposed = sax.fit_transform(ts1)
        >> print(ts1_decomposed)
        >> ts3 = np.vstack((ts1, ts2))
        >> ts3_decomposed = np.apply_along_axis(sax.fit_transform, 1, ts3)
        >> print(ts3_decomposed)

        To obtain encodings that are comparable across series, learn the
        breakpoints once and reuse them:

        >> sax.fit(np.concatenate([ts1, ts2]))
        >> print(sax.transform(ts1), sax.transform(ts2))
        """
        if not isinstance(levels, (list, tuple, np.ndarray)):
            raise TypeError("levels must be a list")
        if not isinstance(bounds, (list, tuple, np.ndarray)):
            raise TypeError("bounds must be a list")
        if isinstance(windows, bool) or not isinstance(windows, (int, np.integer)):
            raise TypeError("windows must be an integer")
        if not isinstance(quantile, (bool, np.bool_)):
            raise TypeError("quantile must be a boolean")

        if len(levels) != (len(bounds) + 1):
            raise ValueError("Length of levels must be equals at length of bounds plus 1")

        if windows < 1:
            raise ValueError("Windows must be a positive integer")

        bounds_array = np.asarray(bounds, dtype=float)
        if np.any(np.diff(bounds_array) <= 0):
            raise ValueError("bounds must be strictly increasing")
        if quantile and ((bounds_array < 0).any() or (bounds_array > 1).any()):
            raise ValueError("with quantile=True the bounds must lie in [0, 1]")

        self.windows = windows
        self.bounds = bounds
        self.levels = levels
        self.quantile = quantile

    def _to_1d_array(self, X):
        if isinstance(X, (list, tuple)):
            X = np.array(X, dtype=float)
        elif isinstance(X, Series):
            X = X.values
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError("X must be a numpy.array or a list or a pandas Series")

        X = np.asarray(X, dtype=float)
        if X.ndim > 1:
            raise TypeError("X must be a 1-D numpy.array")
        return X

    def _paa(self, X):
        """Piecewise Aggregate Approximation: the mean of each time window."""
        if X.shape[0] == 0:
            return np.empty(0)
        # NaN inside a window are ignored; a window made only of NaN stays NaN
        # and is rejected downstream rather than silently dropped
        with np.errstate(invalid="ignore"):
            return np.array([np.nanmean(X[i:i + self.windows]) if not np.all(np.isnan(X[i:i + self.windows])) else np.nan
                             for i in range(0, X.shape[0], self.windows)])

    def fit(self, X, y=None):
        """
        Learn the breakpoints separating the levels.

        Parameters
        --------------------
        X: array-like of shape (1, n_timeSteps)

        Return
        --------------------
        self
        """
        paa = self._paa(self._to_1d_array(X))
        if self.quantile:
            if paa.shape[0] == 0:
                self.breakpoints_ = np.asarray(self.bounds, dtype=float)
            else:
                # nanquantile so that a few missing values do not poison every
                # threshold at once
                self.breakpoints_ = np.nanquantile(paa, np.asarray(self.bounds, dtype=float))
        else:
            self.breakpoints_ = np.asarray(self.bounds, dtype=float)
        return self

    def transform(self, X):
        """
        Encode a series using the breakpoints learned by fit.

        Parameters
        --------------------
        X: array-like of shape (1, n_timeSteps)

        Return
        --------------------
        sax_string: str
            the encoded series, one character per time window.
        """
        if not hasattr(self, "breakpoints_"):
            raise ValueError("this NaiveSAX instance is not fitted yet, call 'fit' first")

        paa = self._paa(self._to_1d_array(X))
        if paa.shape[0] == 0:
            return ''
        if np.isnan(paa).any():
            raise ValueError(
                "a time window contains only missing values and cannot be encoded, "
                "impute the series first (see pynuTS.impute.TsImputer)"
            )

        # searchsorted with side='right' reproduces the original binning rule:
        # level 0 for v < bounds[0], level k for bounds[k-1] <= v < bounds[k],
        # last level for v >= bounds[-1]
        indexes = np.searchsorted(self.breakpoints_, paa, side='right')
        return ''.join(self.levels[i] for i in indexes)
