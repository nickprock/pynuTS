"""
Created on Thu Jun 18 2020

@project: pynuTS
@author: nicola procopio
@last_update: 03/08/2026
@description: Dynamic Time Warping (deprecated, see pynuTS.dtw)
@references: https://iaml.it/blog/serie-storiche-3-dynamic-time-warping
"""

import warnings

import numpy as np

from .dtw import dtw_matrix


def naive_dtw(ts1, ts2, w: int = 1):
    """
    Calculates the distance between two time series using the Dynamic Time Warping.

    .. deprecated:: 0.3.0
        Use :func:`pynuTS.dtw.dtw_distance` or :func:`pynuTS.dtw.dtw_matrix`
        instead. This wrapper is kept only for backward compatibility and
        flattens its inputs to accept the historical ``ndmin=2`` row-vector
        convention.

    Parameters
    -----------------------
    ts1, ts2 : 2D numpy array
        row vectors, as built by ``np.array([...], ndmin=2)``
    w : int.
        default 1. Window parameter (half-width of the Sakoe-Chiba band)

    Returns
    -----------------------
    dist : float.
        The distance between the time series
    DTW_matrix : numpy matrix (or 2D array)
        Distance matrix with warping path.

    Exemple
    -----------------------
    >> import numpy as np
    >> serie_1 = np.array([1, 2, 3, 5, 5, 5, 6], ndmin = 2)
    >> serie_2 = np.array([1, 1, 2, 2, 3, 5], ndmin =2)
    >> from pynuTS.naive_dtw import naive_dtw
    >> dist, DTW_matrix = naive_dtw(ts1 = serie_1, ts2 = serie_2, w=1)
    """
    warnings.warn(
        "pynuTS.naive_dtw.naive_dtw is deprecated since 0.3.0 and will be removed "
        "in a future release, use pynuTS.dtw.dtw_distance instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return dtw_matrix(np.ravel(np.asarray(ts1)), np.ravel(np.asarray(ts2)), w=w)
