"""
Created on Mon Aug 03 2026

@project: pynuTS
@author: nicola procopio
@description: SAX encoding as defined in the original paper
@reference: Lin, Keogh, Lonardi, Chiu (2003), "A symbolic representation of
            time series, with implications for streaming algorithms", DMKD

The difference with :class:`pynuTS.decomposition.NaiveSAX` is not cosmetic.
NaiveSAX cuts the series on its own empirical quantiles, so the alphabet means
something different for every series and the encodings cannot be compared.
Canonical SAX z-normalizes first and then cuts on *fixed* gaussian breakpoints,
which are the same for everybody. That is what buys the property the whole
technique is built on:

    MINDIST(SAX(x), SAX(y))  <=  euclidean_distance(znorm(x), znorm(y))

A distance on the short symbolic strings that never overestimates the distance
on the raw series. Discard a candidate because its MINDIST is already too large
and you know you are not throwing away a true match.
"""

import math
import string

import numpy as np
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["SAX", "gaussian_breakpoints", "norm_ppf", "paa", "znorm"]

# Acklam's rational approximation of the inverse normal CDF
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def norm_ppf(p):
    """
    Inverse of the standard normal cumulative distribution function.

    Implemented here rather than taken from ``scipy.stats.norm.ppf`` only to
    keep scipy out of the dependencies. Acklam's rational approximation gives
    about 1e-9 accuracy and a single Halley step refines it: measured against
    scipy the absolute error stays around 2e-15 over (1e-10, 1 - 1e-6). It grows
    to about 3e-9 for p within 1e-10 of 1, where the culprit is the float
    representation of p itself rather than the algorithm. SAX only ever asks for
    p between 1/26 and 25/26, where the error is at machine precision.

    Parameters
    -----------------------
    p : float or array-like
        probabilities, strictly between 0 and 1.

    Returns
    -----------------------
    x : float or numpy array
        the values such that Phi(x) == p.
    """
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("probabilities must lie strictly between 0 and 1")

    x = np.empty_like(p)

    lower, upper = p < _P_LOW, p > 1 - _P_LOW
    central = ~(lower | upper)

    q = p[central] - 0.5
    r = q * q
    x[central] = ((((((_A[0]*r + _A[1])*r + _A[2])*r + _A[3])*r + _A[4])*r + _A[5]) * q /
                  (((((_B[0]*r + _B[1])*r + _B[2])*r + _B[3])*r + _B[4])*r + 1))

    for mask, tail, sign in ((lower, p[lower], 1.0), (upper, 1 - p[upper], -1.0)):
        if not np.any(mask):
            continue
        q = np.sqrt(-2 * np.log(tail))
        x[mask] = sign * ((((((_C[0]*q + _C[1])*q + _C[2])*q + _C[3])*q + _C[4])*q + _C[5]) /
                          ((((_D[0]*q + _D[1])*q + _D[2])*q + _D[3])*q + 1))

    # one Halley refinement, using the exact CDF through erfc
    erfc = np.vectorize(math.erfc)
    error = 0.5 * erfc(-x / math.sqrt(2)) - p
    u = error * math.sqrt(2 * math.pi) * np.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)

    return x


def gaussian_breakpoints(alphabet_size: int):
    """
    The alphabet_size - 1 cut points that split a standard normal into equally
    probable slices.

    Using equiprobable slices is what makes every symbol equally likely on
    normally distributed data, so no symbol carries more information than
    another.

    Parameters
    -----------------------
    alphabet_size : int
        at least 2.

    Returns
    -----------------------
    breakpoints : 1D numpy array of length alphabet_size - 1

    Example
    -----------------------
    >> from pynuTS.sax import gaussian_breakpoints
    >> gaussian_breakpoints(3)
    array([-0.4307273,  0.4307273])
    """
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least equal to 2")
    return norm_ppf(np.arange(1, alphabet_size) / alphabet_size)


def znorm(X, eps: float = 1e-8):
    """
    Standardize a series to zero mean and unit variance.

    SAX compares *shapes*, so the offset and the scale of a series have to go
    before anything else happens. A series that is constant, or nearly so, has
    no shape to speak of and is mapped to all zeros instead of blowing up.

    Parameters
    -----------------------
    X : 1D array-like
    eps : float
        default 1e-8. Standard deviations below this are treated as zero.

    Returns
    -----------------------
    normalized : 1D numpy array
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] == 0 or np.all(np.isnan(X)):
        # nothing to standardize, and asking numpy for the mean of an empty
        # slice only earns a RuntimeWarning
        return np.zeros_like(X)
    std = np.nanstd(X)
    if std < eps:
        return np.zeros_like(X)
    return (X - np.nanmean(X)) / std


def paa(X, windows: int):
    """
    Piecewise Aggregate Approximation: the mean of each time window.

    Parameters
    -----------------------
    X : 1D array-like
    windows : int
        number of raw points per segment.

    Returns
    -----------------------
    means : 1D numpy array
    lengths : 1D numpy array of int
        how many points went into each segment. All equal to windows except,
        possibly, the last one.
    """
    X = np.asarray(X, dtype=float)
    if windows < 1:
        raise ValueError("windows must be a positive integer")
    if X.shape[0] == 0:
        return np.empty(0), np.empty(0, dtype=int)
    segments = [X[i:i + windows] for i in range(0, X.shape[0], windows)]
    # a window made only of NaN stays NaN and is rejected by the caller, rather
    # than going through nanmean and earning a RuntimeWarning on the way
    means = np.array([np.nan if np.all(np.isnan(s)) else np.nanmean(s) for s in segments])
    lengths = np.array([s.shape[0] for s in segments], dtype=int)
    return means, lengths


class SAX(BaseEstimator, TransformerMixin):
    """
    SAX encoding with the lower-bounding guarantee of the original paper.

    Parameters
    -----------------------
    alphabet : int or list
        default 3. Either the number of symbols, in which case 'a', 'b', 'c'...
        are used, or the explicit list of symbols.
    windows : int
        default 4. Number of raw points aggregated into one symbol (PAA).
    znormalize : bool
        default True. Standardize each series before encoding. Turning this off
        voids the lower-bounding guarantee, it is offered only to inspect what
        the normalization is doing.

    Attributes
    -----------------------
    breakpoints_ : 1D numpy array
        the gaussian cut points, available after fit. They do not depend on the
        data: that is exactly why encodings are comparable across series.

    Example
    -----------------------
    >> import numpy as np
    >> from pynuTS.sax import SAX
    >> sax = SAX(alphabet=4, windows=25)
    >> a = np.sin(np.linspace(0, 6, 100))
    >> b = np.sin(np.linspace(0, 6, 100)) + 100      # same shape, different level
    >> sax.fit(a)
    >> sax.transform(a) == sax.transform(b)
    True
    >> sax.mindist(sax.transform(a), sax.transform(b), n=100)
    0.0
    """

    def __init__(self, alphabet=3, windows: int = 4, znormalize: bool = True):
        if isinstance(alphabet, bool) or not isinstance(alphabet, (int, np.integer, list, tuple)):
            raise TypeError("alphabet must be an integer or a list of symbols")
        if isinstance(alphabet, (int, np.integer)):
            if alphabet < 2:
                raise ValueError("alphabet must contain at least 2 symbols")
            if alphabet > 26:
                raise ValueError("with more than 26 symbols pass the list of symbols explicitly")
        elif len(alphabet) < 2:
            raise ValueError("alphabet must contain at least 2 symbols")
        elif len(set(alphabet)) != len(alphabet):
            raise ValueError("the symbols of the alphabet must be distinct")

        if isinstance(windows, bool) or not isinstance(windows, (int, np.integer)):
            raise TypeError("windows must be an integer")
        if windows < 1:
            raise ValueError("windows must be a positive integer")
        if not isinstance(znormalize, (bool, np.bool_)):
            raise TypeError("znormalize must be a boolean")

        self.alphabet = alphabet
        self.windows = windows
        self.znormalize = znormalize

    @property
    def _symbols(self):
        if isinstance(self.alphabet, (int, np.integer)):
            return list(string.ascii_lowercase[:self.alphabet])
        return list(self.alphabet)

    def fit(self, X=None, y=None):
        """
        Set the gaussian breakpoints.

        The data is not looked at: the breakpoints of canonical SAX are fixed by
        the alphabet size alone. ``fit`` exists so that the class behaves like
        any other scikit-learn transformer.
        """
        self.breakpoints_ = gaussian_breakpoints(len(self._symbols))
        return self

    def transform(self, X):
        """
        Encode one series, or a batch of them.

        Parameters
        -----------------------
        X : 1D array-like, or 2D array-like of shape (n_series, n_timesteps)

        Returns
        -----------------------
        encoded : str, or list of str when X is 2-D
        """
        if not hasattr(self, "breakpoints_"):
            raise ValueError("this SAX instance is not fitted yet, call 'fit' first")

        if isinstance(X, Series):
            X = X.values
        X = np.asarray(X, dtype=float)

        if X.ndim == 2:
            return [self._encode(row) for row in X]
        if X.ndim == 1:
            return self._encode(X)
        raise TypeError("X must be a 1-D or 2-D array")

    def _encode(self, X):
        if self.znormalize:
            X = znorm(X)
        means, _ = paa(X, self.windows)
        if means.shape[0] == 0:
            return ''
        if np.isnan(means).any():
            raise ValueError(
                "a time window contains only missing values and cannot be encoded, "
                "impute the series first (see pynuTS.impute.TsImputer)"
            )
        symbols = self._symbols
        return ''.join(symbols[i] for i in np.searchsorted(self.breakpoints_, means, side='right'))

    def mindist(self, word_a: str, word_b: str, n: int) -> float:
        """
        The lower-bounding distance between two SAX words.

        Guarantees ``mindist(a, b, n) <= euclidean(znorm(A), znorm(B))`` for the
        series A and B the words came from. Symbols that are adjacent in the
        alphabet contribute nothing, because the two segments could in principle
        sit right next to each other on the two sides of a breakpoint.

        Parameters
        -----------------------
        word_a, word_b : str
            two encodings of the same length, produced with the same estimator.
        n : int
            length of the original series, needed to weight each segment by the
            number of raw points behind it.

        Returns
        -----------------------
        distance : float
        """
        if not hasattr(self, "breakpoints_"):
            raise ValueError("this SAX instance is not fitted yet, call 'fit' first")
        if len(word_a) != len(word_b):
            raise ValueError("the two words must have the same length")
        if len(word_a) == 0:
            return 0.0

        index = {symbol: position for position, symbol in enumerate(self._symbols)}
        try:
            a = np.array([index[s] for s in word_a])
            b = np.array([index[s] for s in word_b])
        except KeyError as unknown:
            raise ValueError("symbol %s does not belong to the alphabet" % unknown) from None

        expected = (n + self.windows - 1) // self.windows
        if len(word_a) != expected:
            raise ValueError(
                "a series of %d points encoded with windows=%d gives %d symbols, got %d"
                % (n, self.windows, expected, len(word_a))
            )
        # each segment is weighted by how many raw points it stands for, which
        # keeps the bound valid even when the last window is a short one
        lengths = np.full(len(word_a), self.windows)
        remainder = n % self.windows
        if remainder:
            lengths[-1] = remainder

        low, high = np.minimum(a, b), np.maximum(a, b)
        # np.where evaluates both branches, so the indexes have to stay in range
        # even on the lanes whose value is about to be thrown away
        last = self.breakpoints_.shape[0] - 1
        lower_edge = self.breakpoints_[np.clip(low, 0, last)]
        upper_edge = self.breakpoints_[np.clip(high - 1, 0, last)]
        # adjacent or identical symbols cannot be told apart: distance 0
        gap = np.where(high - low <= 1, 0.0, upper_edge - lower_edge)
        return float(np.sqrt(np.sum(lengths * gap ** 2)))
