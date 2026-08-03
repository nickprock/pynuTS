"""
Created on Mon Aug 03 2026

@project: pynuTS
@author: nicola procopio
@description: DTW Barycenter Averaging
@references: Petitjean, Ketterlin, Gancarski (2011), "A global averaging method
             for dynamic time warping, with applications to clustering",
             Pattern Recognition 44(3)

Averaging time series with the arithmetic mean assumes the points that share an
index also correspond to each other. Under DTW they do not: that is the whole
point of warping. Averaging two identical peaks that are shifted in time gives
a mean with two half-height bumps, a shape neither of the inputs ever had.

DBA fixes this by averaging along the warping path instead of along the index.
"""

import numpy as np

from .dtw import _as_2d, dtw_distance, dtw_path

__all__ = ["dba", "medoid_index"]


def _to_arrays(series):
    arrays = [_as_2d(s) for s in series]
    if len(arrays) == 0:
        raise ValueError("cannot average an empty set of series")
    n_features = arrays[0].shape[1]
    if any(a.shape[1] != n_features for a in arrays):
        raise ValueError("all the series must have the same number of features")
    return arrays


def medoid_index(series, w: int = None, criterion: str = "sqeuclidean") -> int:
    """
    Index of the series minimizing the sum of squared DTW distances to the others.

    It is the recommended starting point for DBA: unlike an arbitrary element it
    already sits in the middle of the group, so the refinement converges faster
    and is less likely to settle on a poor local optimum.

    Parameters
    -----------------------
    series : a list of array-like
    w : int or None
        default None. Half-width of the Sakoe-Chiba band.
    criterion : str
        default 'sqeuclidean'. See :func:`dba`.

    Returns
    -----------------------
    index : int
    """
    arrays = _to_arrays(series)
    best_index, best_cost = 0, float("inf")
    for i, a in enumerate(arrays):
        cost = sum(dtw_distance(a, b, w=w, criterion=criterion) ** 2
                   for j, b in enumerate(arrays) if j != i)
        if cost < best_cost:
            best_index, best_cost = i, cost
    return best_index


def dba(series, init=None, max_iter: int = 10, tol: float = 1e-5,
        w: int = None, criterion: str = "sqeuclidean"):
    """
    DTW Barycenter Averaging: the average of a group of time series under DTW.

    Each pass aligns the current barycenter to every series, collects the points
    each barycenter coordinate is aligned to, and replaces that coordinate with
    their mean. The procedure does not increase the within-group sum of squared
    DTW distances, so it converges.

    The descent is guaranteed only for ``criterion='sqeuclidean'``, which is why
    it is the default here even though the rest of the library defaults to
    'euclidean': replacing a coordinate with the mean of the points aligned to
    it is exactly the step that minimizes the *squared* error, and the path has
    to be optimal for that same quantity for the argument to close.

    Parameters
    -----------------------
    series : a list of array-like
        1-D (univariate) or 2-D of shape (n_timesteps, n_features). The series
        may have different lengths.
    init : array-like or None
        default None. Starting barycenter, which also fixes the output length.
        When None the medoid of the group is used.
    max_iter : int
        default 10. Maximum number of refinement passes.
    tol : float
        default 1e-5. Stop as soon as no coordinate moves more than this.
    w : int or None
        default None. Half-width of the Sakoe-Chiba band.
    criterion : str
        default 'sqeuclidean', the only value for which the descent is
        guaranteed. 'euclidean' and 'cosine' are accepted but then each pass
        only approximately improves the barycenter.

    Returns
    -----------------------
    barycenter : numpy array
        1-D if the inputs are univariate, otherwise 2-D of shape
        (len(init), n_features).

    Example
    -----------------------
    >> import numpy as np
    >> from pynuTS.barycenter import dba
    >> peak = np.exp(-np.linspace(-3, 3, 50) ** 2)
    >> shifted = [np.roll(peak, k) for k in (-6, -2, 2, 6)]
    >> # the arithmetic mean smears the peak, DBA keeps it
    >> print(np.mean(shifted, axis=0).max(), dba(shifted).max())
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least equal to 1")

    arrays = _to_arrays(series)
    univariate = np.asarray(series[0]).ndim == 1

    if init is None:
        barycenter = arrays[medoid_index(arrays, w=w, criterion=criterion)].astype(float).copy()
    else:
        barycenter = _as_2d(init).astype(float).copy()

    if barycenter.shape[1] != arrays[0].shape[1]:
        raise ValueError("init must have the same number of features as the series")

    length = barycenter.shape[0]
    for _ in range(max_iter):
        sums = np.zeros_like(barycenter)
        counts = np.zeros(length)

        for s in arrays:
            _, path = dtw_path(barycenter, s, w=w, criterion=criterion)
            rows = np.fromiter((i for i, _ in path), dtype=int, count=len(path))
            cols = np.fromiter((j for _, j in path), dtype=int, count=len(path))
            # every barycenter coordinate appears at least once in a warping
            # path, so no count can be zero
            counts += np.bincount(rows, minlength=length)
            for feature in range(barycenter.shape[1]):
                sums[:, feature] += np.bincount(rows, weights=s[cols, feature], minlength=length)

        updated = sums / counts[:, None]
        shift = np.abs(updated - barycenter).max()
        barycenter = updated
        if shift <= tol:
            break

    return barycenter.ravel() if univariate else barycenter
