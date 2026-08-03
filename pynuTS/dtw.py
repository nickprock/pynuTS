"""
Created on Thu Jun 18 2020

@project: pynuTS
@author: nicola procopio
@last_update: 03/08/2026
@description: Dynamic Time Warping
@references: https://iaml.it/blog/serie-storiche-3-dynamic-time-warping

This module replaces the external `dtw` package, which is no longer
installable on Python 3.12+, and supersedes the old `pynuTS.naive_dtw`.
"""

import numpy as np

__all__ = ["dtw_distance", "dtw_matrix", "local_cost_matrix"]

_CRITERIA = ("euclidean", "cosine")


def _as_2d(ts):
    """Return the series as a 2-D array of shape (n_timesteps, n_features)."""
    a = np.asarray(ts, dtype=float)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError("a time series must be 1-D (univariate) or 2-D (multivariate), got %d dimensions" % a.ndim)


def local_cost_matrix(ts1, ts2, criterion: str = "euclidean"):
    """
    Point-to-point distance between every element of ts1 and every element of ts2.

    Parameters
    -----------------------
    ts1, ts2 : array-like
        1-D (univariate) or 2-D of shape (n_timesteps, n_features) (multivariate).
    criterion : str
        default 'euclidean'. Either 'euclidean' or 'cosine'.

    Returns
    -----------------------
    cost : 2D numpy array of shape (len(ts1), len(ts2))
    """
    if criterion not in _CRITERIA:
        raise ValueError("criterion must be one of %s, got %r" % (list(_CRITERIA), criterion))

    x, y = _as_2d(ts1), _as_2d(ts2)
    if x.shape[1] != y.shape[1]:
        raise ValueError("the two series must have the same number of features, got %d and %d" % (x.shape[1], y.shape[1]))

    if criterion == "euclidean":
        # broadcasting over the feature axis, then L2 norm
        return np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1))

    # cosine: 1 - similarity. Note that on univariate series every point is a
    # 1-dimensional vector, so the result degenerates to 0.0 (same sign) or
    # 2.0 (opposite sign): 'cosine' is only meaningful on multivariate series.
    norm_x = np.linalg.norm(x, axis=1)
    norm_y = np.linalg.norm(y, axis=1)
    denom = norm_x[:, None] * norm_y[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cost = 1.0 - (x @ y.T) / denom
    # a null vector has no direction: treat it as maximally distant
    return np.where(denom == 0, 1.0, cost)


def dtw_matrix(ts1, ts2, w: int = None, criterion: str = "euclidean"):
    """
    Calculates the distance between two time series using Dynamic Time Warping,
    returning the full accumulated cost matrix.

    Parameters
    -----------------------
    ts1, ts2 : array-like
        1-D (univariate) or 2-D of shape (n_timesteps, n_features) (multivariate).
        The two series may have different lengths.
    w : int or None
        default None. Half-width of the Sakoe-Chiba band: the warping path is
        constrained to |i - j| <= w. It is automatically widened to
        abs(len(ts1) - len(ts2)) when needed, otherwise no path would exist.
        None means no constraint.
    criterion : str
        default 'euclidean'. Either 'euclidean' or 'cosine'.

    Returns
    -----------------------
    dist : float.
        The distance between the time series.
    DTW : 2D numpy array of shape (len(ts1) + 1, len(ts2) + 1)
        Accumulated cost matrix. Cells outside the band hold np.inf.
        DTW[0, 0] is 0.0 and DTW[-1, -1] is the returned distance.

    Example
    -----------------------
    >> import numpy as np
    >> from pynuTS.dtw import dtw_matrix
    >> serie_1 = np.array([1, 2, 3, 5, 5, 5, 6])
    >> serie_2 = np.array([1, 1, 2, 2, 3, 5])
    >> dist, DTW = dtw_matrix(serie_1, serie_2, w=1)
    """
    cost = local_cost_matrix(ts1, ts2, criterion=criterion)
    n, m = cost.shape

    if n == 0 or m == 0:
        raise ValueError("cannot compute DTW on an empty series")

    if w is None:
        band = max(n, m)
    else:
        if w < 1:
            raise ValueError("w must be at least equal to 1")
        # a band narrower than the length difference would leave no feasible path
        band = max(int(w), abs(n - m))

    # the borders must be infinite, otherwise the path could "jump" into the
    # matrix at no cost and the resulting distance would be wrong
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0.0

    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        if lo > hi:
            continue
        prev = DTW[i - 1]
        row = DTW[i]
        cost_i = cost[i - 1]
        # the two predecessors living on the previous row do not depend on the
        # cells we are about to write, so they are computed in one shot
        from_prev_row = np.minimum(prev[lo:hi + 1], prev[lo - 1:hi])
        # the remaining predecessor (same row, previous column) is inherently
        # sequential: this is the only part that has to stay a Python loop
        for offset, j in enumerate(range(lo, hi + 1)):
            row[j] = cost_i[j - 1] + min(from_prev_row[offset], row[j - 1])

    return DTW[n, m], DTW


def dtw_distance(ts1, ts2, w: int = None, criterion: str = "euclidean") -> float:
    """
    Calculates the distance between two time series using Dynamic Time Warping.

    Thin wrapper around :func:`dtw_matrix` for when the accumulated cost matrix
    is not needed.

    Parameters
    -----------------------
    ts1, ts2 : array-like
        1-D (univariate) or 2-D of shape (n_timesteps, n_features) (multivariate).
    w : int or None
        default None. Half-width of the Sakoe-Chiba band, see :func:`dtw_matrix`.
    criterion : str
        default 'euclidean'. Either 'euclidean' or 'cosine'.

    Returns
    -----------------------
    dist : float.
        The distance between the time series.

    Example
    -----------------------
    >> import numpy as np
    >> from pynuTS.dtw import dtw_distance
    >> dtw_distance(np.array([1, 2, 3, 5, 5, 5, 6]), np.array([1, 1, 2, 2, 3, 5]))
    """
    return dtw_matrix(ts1, ts2, w=w, criterion=criterion)[0]
