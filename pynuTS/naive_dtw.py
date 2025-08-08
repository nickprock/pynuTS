"""
Created on Thu Jun 18 2020

@project: pynuTS
@author: nicola procopio
@last_update: 05/08/2025
@description: Dynamic Time Warping
@references: https://iaml.it/blog/serie-storiche-3-dynamic-time-warping
"""

import numpy as np
from typing import Tuple, Optional, Union, Callable, List
import warnings
from scipy.spatial.distance import euclidean, cityblock, minkowski
from numba import jit, prange
import sys

# Try to import numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a dummy decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class DTWCalculator:
    """
    Scalable and optimized Dynamic Time Warping implementation.
    
    This class provides multiple DTW algorithms with different optimizations
    for various use cases and performance requirements.
    
    Parameters
    ----------
    distance_metric : str or callable, default='euclidean'
        Distance metric to use. Options: 'euclidean', 'manhattan', 'chebyshev', 
        'minkowski', or custom callable.
    window_type : str, default='sakoe_chiba'
        Type of constraint window. Options: 'sakoe_chiba', 'itakura', 'no_constraint'.
    p : float, default=2
        Parameter for minkowski distance.
    
    Attributes
    ----------
    _supported_metrics : dict
        Dictionary of supported distance metrics.
    """
    
    def __init__(self, 
                 distance_metric: Union[str, Callable] = 'euclidean',
                 window_type: str = 'sakoe_chiba',
                 p: float = 2):
        
        self.distance_metric = distance_metric
        self.window_type = window_type
        self.p = p
        
        self._supported_metrics = {
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'chebyshev': self._chebyshev_distance,
            'minkowski': self._minkowski_distance
        }
        
        self._distance_func = self._get_distance_function()
    
    def _get_distance_function(self) -> Callable:
        """Get the appropriate distance function."""
        if callable(self.distance_metric):
            return self.distance_metric
        elif self.distance_metric in self._supported_metrics:
            return self._supported_metrics[self.distance_metric]
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    @staticmethod
    def _euclidean_distance(x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> float:
        """Euclidean distance between two points."""
        return np.sqrt(np.sum((x - y) ** 2))
    
    @staticmethod 
    def _manhattan_distance(x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> float:
        """Manhattan distance between two points."""
        return np.sum(np.abs(x - y))
    
    @staticmethod
    def _chebyshev_distance(x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> float:
        """Chebyshev distance between two points."""
        return np.max(np.abs(x - y))
    
    def _minkowski_distance(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> float:
        """Minkowski distance between two points."""
        return np.sum(np.abs(x - y) ** self.p) ** (1.0 / self.p)
    
    def _validate_input(self, ts1: np.ndarray, ts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and prepare input time series."""
        ts1 = np.asarray(ts1)
        ts2 = np.asarray(ts2)
        
        # Handle different input formats
        if ts1.ndim == 1:
            ts1 = ts1.reshape(1, -1)
        elif ts1.ndim > 2:
            raise ValueError("ts1 must be 1D or 2D array")
            
        if ts2.ndim == 1:
            ts2 = ts2.reshape(1, -1)
        elif ts2.ndim > 2:
            raise ValueError("ts2 must be 1D or 2D array")
        
        if ts1.shape[0] != ts2.shape[0]:
            raise ValueError("Time series must have the same number of dimensions")
        
        if ts1.shape[1] == 0 or ts2.shape[1] == 0:
            raise ValueError("Time series cannot be empty")
        
        return ts1, ts2
    
    def _get_window_constraints(self, n: int, m: int, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get window constraints for different constraint types.
        
        Returns
        -------
        lower_bounds, upper_bounds : np.ndarray
            Lower and upper bounds for each row in the DTW matrix.
        """
        if self.window_type == 'sakoe_chiba':
            # Sakoe-Chiba band
            lower_bounds = np.maximum(1, np.arange(1, n+1) - window_size)
            upper_bounds = np.minimum(m, np.arange(1, n+1) + window_size)
            
        elif self.window_type == 'itakura':
            # Itakura parallelogram (simplified version)
            slope = m / n
            lower_bounds = np.maximum(1, np.floor(slope * np.arange(1, n+1) - window_size).astype(int))
            upper_bounds = np.minimum(m, np.ceil(slope * np.arange(1, n+1) + window_size).astype(int))
            
        elif self.window_type == 'no_constraint':
            # No constraints
            lower_bounds = np.ones(n, dtype=int)
            upper_bounds = np.full(n, m, dtype=int)
            
        else:
            raise ValueError(f"Unknown window type: {self.window_type}")
        
        return lower_bounds, upper_bounds
    
    def naive_dtw(self, ts1: np.ndarray, ts2: np.ndarray, 
                  window_size: int = 1) -> Tuple[float, np.ndarray]:
        """
        Original DTW implementation with fixes and optimizations.
        
        Parameters
        ----------
        ts1, ts2 : np.ndarray
            Time series to compare. Can be 1D or 2D arrays.
        window_size : int, default=1
            Window constraint size.
            
        Returns
        -------
        distance : float
            DTW distance between the time series.
        dtw_matrix : np.ndarray
            Full DTW cost matrix.
        """
        ts1, ts2 = self._validate_input(ts1, ts2)
        
        n, m = ts1.shape[1], ts2.shape[1]
        
        # Initialize DTW matrix with infinity
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Adjust window size
        window_size = max(window_size, abs(n - m))
        
        # Get window constraints
        lower_bounds, upper_bounds = self._get_window_constraints(n, m, window_size)
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            j_start = max(1, lower_bounds[i-1])
            j_end = min(m + 1, upper_bounds[i-1] + 1)
            
            for j in range(j_start, j_end):
                # Calculate distance between points
                if ts1.shape[0] == 1:
                    # 1D case - direct subtraction (fixed from original)
                    cost = abs(ts1[0, i-1] - ts2[0, j-1])
                else:
                    # Multi-dimensional case
                    cost = self._distance_func(ts1[:, i-1], ts2[:, j-1])
                
                # DTW recurrence relation
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion  
                    dtw_matrix[i-1, j-1]     # match
                )
        
        return dtw_matrix[n, m], dtw_matrix
    
    def memory_efficient_dtw(self, ts1: np.ndarray, ts2: np.ndarray,
                           window_size: int = 1) -> float:
        """
        Memory-efficient DTW that only keeps two rows in memory.
        Suitable for very long time series.
        
        Parameters
        ----------
        ts1, ts2 : np.ndarray
            Time series to compare.
        window_size : int, default=1
            Window constraint size.
            
        Returns
        -------
        distance : float
            DTW distance between the time series.
        """
        ts1, ts2 = self._validate_input(ts1, ts2)
        
        n, m = ts1.shape[1], ts2.shape[1]
        window_size = max(window_size, abs(n - m))
        
        # Only keep current and previous rows
        prev_row = np.full(m + 1, np.inf)
        curr_row = np.full(m + 1, np.inf)
        prev_row[0] = 0
        
        lower_bounds, upper_bounds = self._get_window_constraints(n, m, window_size)
        
        for i in range(1, n + 1):
            curr_row.fill(np.inf)
            j_start = max(1, lower_bounds[i-1])
            j_end = min(m + 1, upper_bounds[i-1] + 1)
            
            for j in range(j_start, j_end):
                if ts1.shape[0] == 1:
                    cost = abs(ts1[0, i-1] - ts2[0, j-1])
                else:
                    cost = self._distance_func(ts1[:, i-1], ts2[:, j-1])
                
                curr_row[j] = cost + min(
                    prev_row[j],           # insertion
                    curr_row[j-1],         # deletion
                    prev_row[j-1]          # match
                )
            
            # Swap rows
            prev_row, curr_row = curr_row, prev_row
        
        return prev_row[m]
    
    def dtw_with_path(self, ts1: np.ndarray, ts2: np.ndarray,
                     window_size: int = 1) -> Tuple[float, List[Tuple[int, int]]]:
        """
        DTW with optimal warping path recovery.
        
        Parameters
        ----------
        ts1, ts2 : np.ndarray
            Time series to compare.
        window_size : int, default=1
            Window constraint size.
            
        Returns
        -------
        distance : float
            DTW distance.
        path : list of tuples
            Optimal warping path as list of (i, j) coordinates.
        """
        distance, dtw_matrix = self.naive_dtw(ts1, ts2, window_size)
        
        # Backtrack to find optimal path
        path = []
        i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
        
        while i > 0 and j > 0:
            path.append((i-1, j-1))  # Convert to 0-based indexing
            
            # Find the minimum predecessor
            candidates = [
                (dtw_matrix[i-1, j-1], (i-1, j-1)),    # diagonal
                (dtw_matrix[i-1, j], (i-1, j)),        # up
                (dtw_matrix[i, j-1], (i, j-1))         # left
            ]
            
            # Choose the path with minimum cost
            _, (i, j) = min(candidates, key=lambda x: x[0])
        
        # Add remaining path
        while i > 0:
            path.append((i-1, j-1))
            i -= 1
        while j > 0:
            path.append((i-1, j-1))
            j -= 1
        
        path.reverse()
        return distance, path
    
    def fast_dtw_approximation(self, ts1: np.ndarray, ts2: np.ndarray,
                              radius: int = 1, max_iterations: int = 10) -> float:
        """
        FastDTW approximation algorithm for very large time series.
        
        Parameters
        ----------
        ts1, ts2 : np.ndarray
            Time series to compare.
        radius : int, default=1
            Search radius for FastDTW.
        max_iterations : int, default=10
            Maximum number of coarse-to-fine iterations.
            
        Returns
        -------
        distance : float
            Approximate DTW distance.
        """
        ts1, ts2 = self._validate_input(ts1, ts2)
        
        def _reduce_by_half(ts):
            """Reduce time series length by half using averaging."""
            if ts.shape[1] <= 2:
                return ts
            
            n = ts.shape[1]
            if n % 2 == 0:
                # Even length - simple reshape and mean
                reshaped = ts.reshape(ts.shape[0], -1, 2)
                return np.mean(reshaped, axis=2)
            else:
                # Odd length - handle last element separately
                even_part = ts[:, :-1].reshape(ts.shape[0], -1, 2)
                reduced_even = np.mean(even_part, axis=2)
                return np.concatenate([reduced_even, ts[:, -1:]], axis=1)
        
        def _expand_path(path, n1, n2):
            """Expand path from reduced resolution to full resolution."""
            expanded = set()
            for (i, j) in path:
                # Map reduced coordinates to full coordinates
                for di in range(2):
                    for dj in range(2):
                        ni, nj = i * 2 + di, j * 2 + dj
                        if ni < n1 and nj < n2:
                            expanded.add((ni, nj))
            return list(expanded)
        
        # Base case - if series are short enough, use exact DTW
        if ts1.shape[1] <= 50 and ts2.shape[1] <= 50:
            return self.memory_efficient_dtw(ts1, ts2, radius)
        
        # Recursive case - reduce resolution and solve
        ts1_reduced = _reduce_by_half(ts1)
        ts2_reduced = _reduce_by_half(ts2)
        
        # Get approximate solution at reduced resolution
        _, path_reduced = self.dtw_with_path(ts1_reduced, ts2_reduced, radius)
        
        # Expand path to full resolution
        expanded_path = _expand_path(path_reduced, ts1.shape[1], ts2.shape[1])
        
        # Refine solution in neighborhood of expanded path
        # (Simplified version - full FastDTW would do constrained DTW here)
        return self.memory_efficient_dtw(ts1, ts2, radius * 2)


# Optimized functions with JIT compilation (if available)
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _jit_dtw_core(ts1: np.ndarray, ts2: np.ndarray, window_size: int) -> float:
        """JIT-compiled core DTW computation for maximum speed."""
        n, m = ts1.shape[1], ts2.shape[1]
        
        # Use only current and previous rows for memory efficiency
        prev_row = np.full(m + 1, np.inf)
        curr_row = np.full(m + 1, np.inf)
        prev_row[0] = 0
        
        window_size = max(window_size, abs(n - m))
        
        for i in range(1, n + 1):
            curr_row[:] = np.inf
            j_start = max(1, i - window_size)
            j_end = min(m + 1, i + window_size + 1)
            
            for j in prange(j_start, j_end):
                cost = abs(ts1[0, i-1] - ts2[0, j-1])
                curr_row[j] = cost + min(
                    prev_row[j],       # insertion
                    curr_row[j-1],     # deletion
                    prev_row[j-1]      # match
                )
            
            # Swap rows
            prev_row, curr_row = curr_row, prev_row
        
        return prev_row[m]


# Convenience functions maintaining backward compatibility
def naive_dtw(ts1: np.ndarray, ts2: np.ndarray, w: int = 1) -> Tuple[float, np.ndarray]:
    """
    Enhanced version of the original naive_dtw function with bug fixes.
    
    Parameters
    ----------
    ts1, ts2 : np.ndarray
        Time series to compare. Can be 1D or 2D arrays.
    w : int, default=1
        Window parameter for constraints.
        
    Returns
    -------
    distance : float
        DTW distance between the time series.
    dtw_matrix : np.ndarray
        Distance matrix with warping information.
        
    Examples
    --------
    >>> import numpy as np
    >>> serie_1 = np.array([1, 2, 3, 5, 5, 5, 6])
    >>> serie_2 = np.array([1, 1, 2, 2, 3, 5])
    >>> dist, DTW_matrix = naive_dtw(serie_1, serie_2, w=1)
    """
    calculator = DTWCalculator()
    return calculator.naive_dtw(ts1, ts2, w)


def fast_dtw(ts1: np.ndarray, ts2: np.ndarray, 
             window_size: int = 1, 
             distance_metric: str = 'euclidean') -> float:
    """
    Fast DTW computation with memory optimization.
    
    Parameters
    ----------
    ts1, ts2 : np.ndarray
        Time series to compare.
    window_size : int, default=1
        Window constraint size.
    distance_metric : str, default='euclidean'
        Distance metric to use.
        
    Returns
    -------
    distance : float
        DTW distance.
    """
    if NUMBA_AVAILABLE and distance_metric == 'euclidean':
        # Use JIT-optimized version for maximum speed
        ts1 = np.asarray(ts1)
        ts2 = np.asarray(ts2)
        if ts1.ndim == 1:
            ts1 = ts1.reshape(1, -1)
        if ts2.ndim == 1:
            ts2 = ts2.reshape(1, -1)
            
        if ts1.shape[0] == 1 and ts2.shape[0] == 1:
            return _jit_dtw_core(ts1, ts2, window_size)
    
    # Fallback to regular implementation
    calculator = DTWCalculator(distance_metric=distance_metric)
    return calculator.memory_efficient_dtw(ts1, ts2, window_size)


def dtw_distance_matrix(time_series_list: List[np.ndarray], 
                       window_size: int = 1,
                       distance_metric: str = 'euclidean',
                       n_jobs: int = 1) -> np.ndarray:
    """
    Compute pairwise DTW distance matrix for multiple time series.
    
    Parameters
    ----------
    time_series_list : list of np.ndarray
        List of time series to compare.
    window_size : int, default=1
        Window constraint size.
    distance_metric : str, default='euclidean'
        Distance metric to use. 
    n_jobs : int, default=1
        Number of parallel jobs (placeholder for future implementation).
        
    Returns
    -------
    distance_matrix : np.ndarray
        Symmetric matrix of pairwise DTW distances.
    """
    n_series = len(time_series_list)
    distance_matrix = np.zeros((n_series, n_series))
    
    calculator = DTWCalculator(distance_metric=distance_metric)
    
    for i in range(n_series):
        for j in range(i + 1, n_series):
            dist = calculator.memory_efficient_dtw(
                time_series_list[i], 
                time_series_list[j], 
                window_size
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric
    
    return distance_matrix


def dtw_barycenter(time_series_list: List[np.ndarray], 
                  max_iterations: int = 10,
                  window_size: int = 1) -> np.ndarray:
    """
    Compute DTW barycenter (average) of multiple time series.
    
    Parameters
    ----------
    time_series_list : list of np.ndarray
        List of time series to average.
    max_iterations : int, default=10
        Maximum number of iterations for the averaging algorithm.
    window_size : int, default=1
        Window constraint size.
        
    Returns
    -------
    barycenter : np.ndarray
        DTW barycenter of the input time series.
    """
    if not time_series_list:
        raise ValueError("time_series_list cannot be empty")
    
    # Initialize with mean length
    lengths = [len(ts) if ts.ndim == 1 else ts.shape[1] for ts in time_series_list]
    target_length = int(np.mean(lengths))
    
    # Initialize barycenter as simple average (interpolated to target length)
    barycenter = np.zeros(target_length)
    
    calculator = DTWCalculator()
    
    for iteration in range(max_iterations):
        # Update barycenter based on DTW alignments
        aligned_series = []
        
        for ts in time_series_list:
            _, path = calculator.dtw_with_path(
                barycenter.reshape(1, -1), 
                ts.reshape(1, -1) if ts.ndim == 1 else ts,
                window_size
            )
            
            # Create aligned version of time series
            aligned = np.zeros(target_length)
            path_dict = {}
            
            for i, j in path:
                if i not in path_dict:
                    path_dict[i] = []
                if ts.ndim == 1:
                    path_dict[i].append(ts[j])
                else:
                    path_dict[i].append(ts[0, j])
            
            for i in range(target_length):
                if i in path_dict:
                    aligned[i] = np.mean(path_dict[i])
                else:
                    # Interpolate missing values
                    if i > 0:
                        aligned[i] = aligned[i-1]
            
            aligned_series.append(aligned)
        
        # Update barycenter
        new_barycenter = np.mean(aligned_series, axis=0)
        
        # Check convergence
        if np.allclose(barycenter, new_barycenter, rtol=1e-6):
            break
            
        barycenter = new_barycenter
    
    return barycenter