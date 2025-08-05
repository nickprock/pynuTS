"""
Created on Wed Apr 01 2020

@project: pynuTS
@author: nicola procopio
@last_update: 05/08/2025
@description: impute missing value with rolling mean
@reference: https://iaml.it/blog/serie-storiche-1-dati-mancanti
"""
import numpy as np
from typing import Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

class TsImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in time series using rolling mean with improved scalability.

    Parameters
    ----------
    m_avg : int, default=1
        The range of the moving average window.
    copy : bool, default=True
        If True, create a copy of input X, otherwise modify in-place.
    min_periods : int, default=1
        Minimum number of observations required to have a value.
    verbose : bool, default=False
        Whether to show progress bar for large datasets.
    chunk_size : int, optional
        Process data in chunks for memory efficiency. Auto-determined if None.
    
    Returns
    -------
    numpy.ndarray
        Imputed time series data.

    Examples
    --------
    >>> import numpy as np
    >>> from improved_impute import TsImputer, maximum_distance_recommended
    >>> X = np.array([1, 2, np.nan, 3, 5, np.nan])
    >>> dist = maximum_distance_recommended(X)
    >>> imputer = TsImputer(m_avg=dist)
    >>> X_new = imputer.fit_transform(X)
    """
    
    def __init__(self, m_avg: int = 1, copy: bool = True, min_periods: int = 1, 
                 verbose: bool = False, chunk_size: Optional[int] = None):
        self._validate_parameters(m_avg, min_periods)
        
        self.m_avg = m_avg
        self.copy = copy
        self.min_periods = min_periods
        self.verbose = verbose
        self.chunk_size = chunk_size
        
    def _validate_parameters(self, m_avg: int, min_periods: int) -> None:
        """Validate input parameters."""
        if m_avg is None or m_avg < 1:
            raise ValueError("m_avg must be a positive integer")
        if min_periods < 1:
            raise ValueError("min_periods must be a positive integer")
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and prepare input data."""
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Input must be a 1D array")
        if len(X) == 0:
            raise ValueError("Input array cannot be empty")
        return X
    
    def _get_window_bounds(self, position: int, data_length: int) -> Tuple[int, int]:
        """Calculate window bounds for moving average."""
        low_bound = max(0, position - self.m_avg)
        upp_bound = min(data_length, position + self.m_avg + 1)
        return low_bound, upp_bound
    
    def _compute_moving_average_vectorized(self, data: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of moving averages for multiple positions.
        More efficient for large datasets.
        """
        n_positions = len(positions)
        results = np.full(n_positions, np.nan)
        
        for idx, pos in enumerate(positions):
            low_bound, upp_bound = self._get_window_bounds(pos, len(data))
            window_data = data[low_bound:upp_bound]
            
            # Filter out NaN values
            valid_data = window_data[~np.isnan(window_data)]
            
            if len(valid_data) >= self.min_periods:
                results[idx] = np.mean(valid_data)
            else:
                # Fallback: use global mean if available
                global_valid = data[~np.isnan(data)]
                if len(global_valid) > 0:
                    results[idx] = np.mean(global_valid)
                else:
                    results[idx] = 0.0  # Last resort
                    
        return results
    
    def _process_chunk(self, data: np.ndarray, na_positions: np.ndarray) -> np.ndarray:
        """Process a chunk of missing value positions."""
        if len(na_positions) == 0:
            return data
            
        # Compute imputed values for all positions at once
        imputed_values = self._compute_moving_average_vectorized(data, na_positions)
        
        # Update the data
        data[na_positions] = imputed_values
        return data
    
    def _determine_chunk_size(self, n_missing: int) -> int:
        """Automatically determine optimal chunk size based on data size."""
        if self.chunk_size is not None:
            return self.chunk_size
            
        # Adaptive chunk sizing
        if n_missing < 1000:
            return n_missing  # Process all at once
        elif n_missing < 10000:
            return 1000
        else:
            return 5000
    
    def fit(self, X: np.ndarray, y=None):
        """Fit the imputer (no-op for this implementation)."""
        X = self._validate_input(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the input by imputing missing values."""
        X = self._validate_input(X)
        
        # Find missing values
        na_mask = np.isnan(X)
        na_positions = np.where(na_mask)[0]
        
        if len(na_positions) == 0:
            return X.copy() if self.copy else X
        
        # Prepare data
        data = X.copy() if self.copy else X
        
        # Determine processing strategy
        chunk_size = self._determine_chunk_size(len(na_positions))
        
        if self.verbose and len(na_positions) > 100:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(na_positions), chunk_size), 
                              desc="Imputing missing values")
            except ImportError:
                iterator = range(0, len(na_positions), chunk_size)
                warnings.warn("tqdm not available, processing without progress bar")
        else:
            iterator = range(0, len(na_positions), chunk_size)
        
        # Process in chunks
        for start_idx in iterator:
            end_idx = min(start_idx + chunk_size, len(na_positions))
            chunk_positions = na_positions[start_idx:end_idx]
            data = self._process_chunk(data, chunk_positions)
        
        return data
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit the imputer and transform the input."""
        return self.fit(X, y).transform(X)


def maximum_distance_recommended(X: np.ndarray, min_gap_threshold: int = 2) -> int:
    """
    Recommend the maximum range for moving average window based on gap analysis.
    
    Parameters
    ----------
    X : numpy.ndarray
        1D time series array.
    min_gap_threshold : int, default=2
        Minimum gap size to consider for recommendations.
    
    Returns
    -------
    int
        Recommended maximum distance for m_avg parameter.
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([1, 2, np.nan, 3, 5, np.nan])
    >>> dist = maximum_distance_recommended(X)
    """
    X = np.asarray(X)
    if X.ndim != 1:
        raise ValueError("Input must be a 1D array")
    
    na_positions = np.where(np.isnan(X))[0]
    
    if len(na_positions) == 0:
        # No missing values - use half the series length
        max_range = max(1, int((len(X) - 1) / 2) - 1)
        print(f"No missing values found. Recommended m_avg parameter: {max_range}")
        return max_range
    
    if len(na_positions) == 1:
        # Single missing value
        max_range = min(na_positions[0], len(X) - na_positions[0] - 1)
        max_range = max(1, max_range)
        print(f"Single missing value. Recommended m_avg parameter: {max_range}")
        return max_range
    
    # Calculate gaps between consecutive missing values
    gaps = np.diff(na_positions)
    
    # Filter gaps that are meaningful
    meaningful_gaps = gaps[gaps >= min_gap_threshold]
    
    if len(meaningful_gaps) == 0:
        # All gaps are too small - use minimum safe value
        max_range = 1
    else:
        # Use the smallest meaningful gap minus 1
        max_range = max(1, int(np.min(meaningful_gaps)) - 1)
    
    print(f"Recommended m_avg parameter: {max_range}")
    return max_range


# Additional utility functions for advanced usage
def impute_with_strategy(X: np.ndarray, strategy: str = 'rolling_mean', **kwargs) -> np.ndarray:
    """
    Convenience function to impute with different strategies.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input time series.
    strategy : str, default='rolling_mean'
        Imputation strategy. Currently supports 'rolling_mean'.
    **kwargs : dict
        Additional parameters for the chosen strategy.
    
    Returns
    -------
    numpy.ndarray
        Imputed time series.
    """
    if strategy == 'rolling_mean':
        imputer = TsImputer(**kwargs)
        return imputer.fit_transform(X)
    else:
        raise ValueError(f"Strategy '{strategy}' not supported")


def analyze_missing_patterns(X: np.ndarray) -> dict:
    """
    Analyze missing value patterns in the time series.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input time series.
    
    Returns
    -------
    dict
        Dictionary with analysis results.
    """
    X = np.asarray(X)
    na_mask = np.isnan(X)
    na_positions = np.where(na_mask)[0]
    
    if len(na_positions) == 0:
        return {
            'total_missing': 0,
            'missing_percentage': 0.0,
            'longest_gap': 0,
            'average_gap': 0.0,
            'gaps': []
        }
    
    # Calculate gaps
    if len(na_positions) > 1:
        gaps = np.diff(na_positions)
        longest_gap = np.max(gaps)
        average_gap = np.mean(gaps)
    else:
        gaps = []
        longest_gap = 0
        average_gap = 0.0
    
    return {
        'total_missing': len(na_positions),
        'missing_percentage': (len(na_positions) / len(X)) * 100,
        'longest_gap': longest_gap,
        'average_gap': average_gap,
        'gaps': gaps.tolist() if len(gaps) > 0 else []
    }