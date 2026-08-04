"""
Created on Tue Aug 04 2026

@project: pynuTS
@author: nicola procopio
@description: mean-scaled uniform quantization, the tokenization scheme used to
              feed time series to a language model
@reference: Ansari et al. (2024), "Chronos: Learning the Language of Time Series"

Chronos, TimesFM and the LLM-based forecasters that appeared from 2023 onwards
all rest on one move: turn a real valued series into a sequence of symbols drawn
from a finite alphabet, then hand it to a model built for language. That is the
same move SAX made in 2003, for a different purpose.

Putting the two side by side is the point of this module:

                    SAX                          this module
    scaling         z-normalization              divide by mean absolute value
    time axis       PAA, w points per symbol     one token per point
    cut points      equiprobable gaussian        uniform over a fixed range
    alphabet        small, 3 to 10 symbols       large, thousands of tokens
    built for       indexing and lower bounds    input to a language model
    invertible      no, heavily lossy            yes, up to the bin width

This implements the *scheme*, not the model: no weights, no vocabulary layout,
no special tokens. Chronos reserves a couple of ids for padding and
end-of-sequence, which is irrelevant to the idea and is left out here.
"""

import numpy as np
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["MeanScaleQuantizer", "mean_scale"]


def mean_scale(X, eps: float = 1e-8):
    """
    Divide a series by the mean of its absolute values.

    Unlike z-normalization this keeps the sign and the zero: a series that is
    always positive stays always positive, which matters when the numbers are
    counts or demand and the model is expected to produce more of the same.

    Parameters
    -----------------------
    X : 1D array-like
    eps : float
        default 1e-8. Scales below this are replaced by 1.0, so an all zero
        series survives instead of turning into NaN.

    Returns
    -----------------------
    scaled : 1D numpy array
    scale : float
        the divisor, needed to bring the values back.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] == 0 or np.all(np.isnan(X)):
        return X.copy(), 1.0
    scale = float(np.nanmean(np.abs(X)))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return X / scale, scale


class MeanScaleQuantizer(BaseEstimator, TransformerMixin):
    """
    Mean scaling followed by uniform binning, the way a time series is turned
    into tokens for a language model.

    Parameters
    -----------------------
    n_bins : int
        default 4096, the order of magnitude Chronos uses. One token per bin.
    low, high : float
        default -15.0 and 15.0. The range the bin centers span, in units of the
        scaled series. Values outside it saturate on the outermost bin, which
        is what keeps a single spike from stretching the whole grid.

    Attributes
    -----------------------
    centers_ : 1D numpy array
        the value each token decodes to, available after fit.
    edges_ : 1D numpy array
        the midpoints between consecutive centers.
    scales_ : 1D numpy array
        the scale used for each series of the last transform.

    Example
    -----------------------
    >> import numpy as np
    >> from pynuTS.quantize import MeanScaleQuantizer
    >> x = np.sin(np.linspace(0, 6, 100)) * 20 + 100
    >> q = MeanScaleQuantizer(n_bins=256).fit()
    >> tokens = q.transform(x)
    >> back = q.inverse_transform(tokens)
    >> float(np.abs(back - x).max())
    """

    def __init__(self, n_bins: int = 4096, low: float = -15.0, high: float = 15.0):
        if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
            raise TypeError("n_bins must be an integer")
        if n_bins < 2:
            raise ValueError("n_bins must be at least equal to 2")
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("low and high must be finite")
        if low >= high:
            raise ValueError("low must be strictly smaller than high")

        self.n_bins = n_bins
        self.low = low
        self.high = high

    def fit(self, X=None, y=None):
        """
        Lay out the grid of bins.

        Like the gaussian breakpoints of :class:`pynuTS.sax.SAX`, the grid does
        not depend on the data: it is fixed once and shared by every series, so
        the same token always decodes to the same scaled value.
        """
        self.centers_ = np.linspace(self.low, self.high, self.n_bins)
        self.edges_ = (self.centers_[1:] + self.centers_[:-1]) / 2
        return self

    def transform(self, X):
        """
        Turn a series, or a batch of them, into token ids.

        Parameters
        -----------------------
        X : 1D array-like, or 2D array-like of shape (n_series, n_timesteps)

        Returns
        -----------------------
        tokens : numpy array of int, same shape as X
        """
        self._check_fitted()
        X, was_1d = self._as_2d(X)

        tokens = np.empty(X.shape, dtype=int)
        scales = np.empty(X.shape[0])
        for row in range(X.shape[0]):
            scaled, scales[row] = mean_scale(X[row])
            tokens[row] = np.searchsorted(self.edges_, scaled, side='right')
        self.scales_ = scales

        return tokens[0] if was_1d else tokens

    def inverse_transform(self, tokens, scale=None):
        """
        Decode token ids back into values.

        Parameters
        -----------------------
        tokens : array-like of int, 1-D or 2-D
        scale : float, array-like or None
            default None, meaning the scales stored by the last transform.

        Returns
        -----------------------
        values : numpy array of float, same shape as tokens
        """
        self._check_fitted()
        tokens = np.asarray(tokens)
        was_1d = tokens.ndim == 1
        tokens = np.atleast_2d(tokens)

        if (tokens < 0).any() or (tokens >= self.n_bins).any():
            raise ValueError("token ids must lie between 0 and n_bins - 1")

        if scale is None:
            if not hasattr(self, "scales_"):
                raise ValueError("no scale available, pass one or call transform first")
            scale = self.scales_
        scale = np.atleast_1d(np.asarray(scale, dtype=float))
        if scale.shape[0] != tokens.shape[0]:
            raise ValueError("expected %d scales, got %d" % (tokens.shape[0], scale.shape[0]))

        values = self.centers_[tokens] * scale[:, None]
        return values[0] if was_1d else values

    def bin_width(self) -> float:
        """The spacing of the grid, which bounds the error of a round trip."""
        self._check_fitted()
        return float(self.centers_[1] - self.centers_[0])

    def _check_fitted(self):
        if not hasattr(self, "centers_"):
            raise ValueError("this MeanScaleQuantizer instance is not fitted yet, call 'fit' first")

    @staticmethod
    def _as_2d(X):
        if isinstance(X, Series):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X.reshape(1, -1), True
        if X.ndim == 2:
            return X, False
        raise TypeError("X must be a 1-D or 2-D array")
