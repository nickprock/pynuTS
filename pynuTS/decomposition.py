"""
Created on Wed May 15 2021

@project: pynuTS
@author: nicola procopio
@last_update: 05/08/2025
@description: dimensionality reduction by SAX encoding
@reference: https://iaml.it/blog/serie-storiche-2-sax-encoding
"""

import numpy as np
from typing import List, Union, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from collections import defaultdict
import warnings

class NaiveSAX(BaseEstimator, TransformerMixin):
    """
    Scalable SAX Encoding (Symbolic Aggregate approXimation) implementation.
    
    SAX allows for dimensionality reduction and indexing with a lower-bounding 
    distance measure. Invented by Eamonn Keogh and Jessica Lin in 2002.
    
    Parameters
    ----------
    levels : list, default=["A", "B", "C"]
        Labels for SAX Encoding symbols.
    bounds : list, default=[0.25, 0.75]  
        Quantile boundaries for binning. With default values: 3 levels
        [(min, 25th percentile), (25th percentile, 75th percentile), (75th percentile, max)]
    windows : int, default=2
        Time window size for PAA (Piecewise Aggregate Approximation).
    quantile : bool, default=True
        If True, use quantile-based boundaries. If False, use absolute values in bounds.
    overlap : bool, default=False
        Whether to use overlapping windows for PAA.
    min_window_size : int, default=1
        Minimum window size for the last window if series length is not divisible by windows.
    cache_quantiles : bool, default=True
        Cache computed quantiles for repeated transformations with same data distribution.
    
    Attributes
    ----------
    _cached_quantiles : dict
        Cached quantile values for performance optimization.
    _is_fitted : bool
        Whether the transformer has been fitted.
    
    Examples
    --------
    >>> import numpy as np
    >>> from improved_sax import NaiveSAX
    >>> ts1 = 2.5 * np.random.randn(100,) + 3
    >>> sax = NaiveSAX(windows=5, levels=['A', 'B', 'C', 'D'])
    >>> ts1_sax = sax.fit_transform(ts1)
    >>> print(ts1_sax)
    
    # Batch processing multiple series
    >>> ts_batch = np.random.randn(10, 100)  # 10 series of length 100
    >>> sax_batch = sax.transform_batch(ts_batch)
    """
    
    def __init__(self, 
                levels: List[str] = None,
                bounds: List[float] = None, 
                windows: int = 2,
                quantile: bool = True,
                overlap: bool = False,
                min_window_size: int = 1,
                cache_quantiles: bool = True):
        
        # Set defaults
        if levels is None:
            levels = ["A", "B", "C"]
        if bounds is None:
            bounds = [0.25, 0.75]

        # Assign FIRST
        self.levels = levels
        self.bounds = np.array(bounds)
        self.windows = windows
        self.quantile = quantile
        self.overlap = overlap
        self.min_window_size = min_window_size
        self.cache_quantiles = cache_quantiles

        # Internal state
        self._cached_quantiles = {}
        self._is_fitted = False

        # THEN validate
        self._validate_parameters(self.levels, self.bounds, self.windows, self.min_window_size)

        
    def _validate_parameters(self, levels: List[str], bounds: List[float], 
                           windows: int, min_window_size: int) -> None:
        """Validate input parameters."""
        if len(levels) != (len(bounds) + 1):
            raise ValueError(
                f"Length of levels ({len(levels)}) must equal length of bounds + 1 ({len(bounds) + 1})"
            )
        
        if windows < 1:
            raise ValueError("Windows must be a positive integer")
            
        if min_window_size < 1:
            raise ValueError("min_window_size must be a positive integer")
            
        if self.quantile and (np.any(np.array(bounds) < 0) or np.any(np.array(bounds) > 1)):
            raise ValueError("When quantile=True, bounds must be between 0 and 1")
            
        if not np.all(np.diff(bounds) > 0):
            raise ValueError("Bounds must be in ascending order")
    
    def _validate_input(self, X: Union[np.ndarray, list, pd.Series]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError(
                "X must be a numpy.array, list, or pandas Series"
            )
        
        if X.ndim > 1:
            raise ValueError("X must be a 1-D array")
            
        if len(X) == 0:
            raise ValueError("Input array cannot be empty")
            
        return X
    
    def _compute_paa_vectorized(self, X: np.ndarray) -> np.ndarray:
        """
        Compute PAA (Piecewise Aggregate Approximation) using vectorized operations.
        Much faster than the original loop-based approach.
        """
        n = len(X)
        
        if self.overlap:
            # Overlapping windows
            if n < self.windows:
                return np.array([np.nanmean(X)])
            
            step = max(1, self.windows // 2)  # 50% overlap
            paa_values = []
            
            for i in range(0, n - self.windows + 1, step):
                window_mean = np.nanmean(X[i:i + self.windows])
                paa_values.append(window_mean)
                
            return np.array(paa_values)
        
        else:
            # Non-overlapping windows (original behavior, optimized)
            if n % self.windows == 0:
                # Perfect division
                reshaped = X.reshape(-1, self.windows)
                return np.nanmean(reshaped, axis=1)
            else:
                # Handle remainder
                n_complete_windows = n // self.windows
                paa_values = []
                
                if n_complete_windows > 0:
                    complete_data = X[:n_complete_windows * self.windows]
                    complete_reshaped = complete_data.reshape(-1, self.windows)
                    paa_values.extend(np.nanmean(complete_reshaped, axis=1))
                
                # Handle remainder window
                remainder_start = n_complete_windows * self.windows
                if remainder_start < n:
                    remainder = X[remainder_start:]
                    if len(remainder) >= self.min_window_size:
                        paa_values.append(np.nanmean(remainder))
                
                return np.array(paa_values)
    
    def _get_quantile_key(self, paa_values: np.ndarray) -> str:
        """Generate a key for caching quantiles based on data characteristics."""
        if not self.cache_quantiles:
            return None
        
        # Use statistical properties as cache key
        stats = (
            len(paa_values),
            float(np.nanmean(paa_values)),
            float(np.nanstd(paa_values)),
            float(np.nanmin(paa_values)),
            float(np.nanmax(paa_values))
        )
        return str(hash(stats))
    
    def _compute_boundaries(self, paa_values: np.ndarray) -> np.ndarray:
        """Compute boundaries for binning, with caching for performance."""
        if not self.quantile:
            return self.bounds
        
        # Try to use cached quantiles
        cache_key = self._get_quantile_key(paa_values)
        if cache_key and cache_key in self._cached_quantiles:
            return self._cached_quantiles[cache_key]
        
        # Compute quantiles
        boundaries = np.quantile(paa_values[~np.isnan(paa_values)], self.bounds)
        
        # Cache if enabled
        if cache_key:
            self._cached_quantiles[cache_key] = boundaries
        
        return boundaries
    
    def _discretize_vectorized(self, paa_values: np.ndarray) -> np.ndarray:
        """
        Vectorized discretization of PAA values into SAX symbols.
        Much faster than the original nested loop approach.
        """
        boundaries = self._compute_boundaries(paa_values)
        
        # Initialize result array
        result = np.full(len(paa_values), self.levels[-1], dtype=object)
        
        # Vectorized binning
        for i, boundary in enumerate(boundaries):
            mask = paa_values < boundary
            result[mask] = self.levels[i]
        
        return result
    
    def fit(self, X: Union[np.ndarray, list, pd.Series], y=None):
        """
        Fit the SAX transformer.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data.
        y : ignored
            Not used, present for sklearn compatibility.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_input(X)
        self._is_fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, list, pd.Series]) -> str:
        """
        Transform time series to SAX representation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data to transform.
            
        Returns
        -------
        sax_string : str
            SAX encoded string representation.
        """
        X = self._validate_input(X)
        
        # Handle edge cases
        if len(X) < self.min_window_size:
            warnings.warn(
                f"Input length ({len(X)}) is smaller than min_window_size ({self.min_window_size}). "
                "Returning single symbol based on mean value."
            )
            # Use global statistics for single value
            if self.quantile:
                # Use middle symbol for short series
                middle_idx = len(self.levels) // 2
                return self.levels[middle_idx]
            else:
                # Simple binning based on bounds
                mean_val = np.nanmean(X)
                for i, bound in enumerate(self.bounds):
                    if mean_val < bound:
                        return self.levels[i]
                return self.levels[-1]
        
        # Compute PAA
        paa_values = self._compute_paa_vectorized(X)
        
        if len(paa_values) == 0:
            return ""
        
        # Discretize to symbols
        symbols = self._discretize_vectorized(paa_values)
        
        # Join symbols into string
        return ''.join(symbols)
    
    def fit_transform(self, X: Union[np.ndarray, list, pd.Series], y=None) -> str:
        """
        Fit the transformer and transform the input.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data.
        y : ignored
            Not used, present for sklearn compatibility.
            
        Returns
        -------
        sax_string : str
            SAX encoded string representation.
        """
        return self.fit(X, y).transform(X)
    
    def transform_batch(self, X_batch: np.ndarray, n_jobs: int = 1) -> List[str]:
        """
        Transform multiple time series efficiently.
        
        Parameters
        ----------
        X_batch : array-like of shape (n_series, n_timepoints)
            Multiple time series to transform.
        n_jobs : int, default=1
            Number of parallel jobs. Currently not implemented (future enhancement).
            
        Returns
        -------
        sax_strings : list of str
            List of SAX encoded strings, one for each input series.
        """
        X_batch = np.asarray(X_batch)
        if X_batch.ndim != 2:
            raise ValueError("X_batch must be a 2D array")
        
        results = []
        for i in range(X_batch.shape[0]):
            try:
                sax_string = self.transform(X_batch[i])
                results.append(sax_string)
            except Exception as e:
                warnings.warn(f"Failed to transform series {i}: {str(e)}")
                results.append("")  # Empty string for failed transformations
        
        return results
    
    def get_paa_values(self, X: Union[np.ndarray, list, pd.Series]) -> np.ndarray:
        """
        Get the PAA (Piecewise Aggregate Approximation) values without discretization.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data.
            
        Returns
        -------
        paa_values : np.ndarray
            PAA values.
        """
        X = self._validate_input(X)
        return self._compute_paa_vectorized(X)
    
    def inverse_transform_approximate(self, sax_string: str, original_length: int) -> np.ndarray:
        """
        Approximate inverse transformation from SAX string back to time series.
        Note: This is an approximation and won't recover the original series exactly.
        
        Parameters
        ----------
        sax_string : str
            SAX encoded string.
        original_length : int
            Target length for the reconstructed series.
            
        Returns
        -------
        reconstructed : np.ndarray
            Approximately reconstructed time series.
        """
        if not sax_string:
            return np.zeros(original_length)
        
        # Map symbols back to approximate values (using midpoints of bins)
        symbol_to_value = {}
        
        # Create dummy boundaries for mapping (simplified approach)
        dummy_boundaries = np.linspace(-2, 2, len(self.levels) + 1)
        
        for i, level in enumerate(self.levels):
            midpoint = (dummy_boundaries[i] + dummy_boundaries[i + 1]) / 2
            symbol_to_value[level] = midpoint
        
        # Convert symbols to values
        symbol_values = [symbol_to_value.get(symbol, 0) for symbol in sax_string]
        
        # Expand to original length
        if len(symbol_values) == 0:
            return np.zeros(original_length)
        
        # Simple linear interpolation to target length
        reconstructed = np.repeat(symbol_values, original_length // len(symbol_values))
        
        # Handle remainder
        remainder = original_length - len(reconstructed)
        if remainder > 0:
            reconstructed = np.concatenate([
                reconstructed, 
                np.repeat(symbol_values[-1], remainder)
            ])
        
        return reconstructed[:original_length]
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for sklearn compatibility."""
        return ["sax_representation"]
        
    def clear_cache(self):
        """Clear the quantile cache."""
        self._cached_quantiles.clear()


# Utility functions for advanced SAX operations
def sax_distance(sax1: str, sax2: str, alphabet_size: int = None) -> float:
    """
    Compute the distance between two SAX strings.
    
    Parameters
    ----------
    sax1, sax2 : str
        SAX encoded strings to compare.
    alphabet_size : int, optional
        Size of the alphabet used. Auto-detected if None.
        
    Returns
    -------
    distance : float
        Distance between the two SAX strings.
    """
    if len(sax1) != len(sax2):
        raise ValueError("SAX strings must have the same length")
    
    if alphabet_size is None:
        # Auto-detect alphabet size
        unique_chars = set(sax1 + sax2)
        alphabet_size = len(unique_chars)
    
    # Simple character-based distance (can be enhanced with proper SAX distance)
    distance = sum(c1 != c2 for c1, c2 in zip(sax1, sax2))
    return distance / len(sax1)  # Normalize


def analyze_sax_patterns(sax_strings: List[str]) -> dict:
    """
    Analyze patterns in a collection of SAX strings.
    
    Parameters
    ----------
    sax_strings : list of str
        Collection of SAX encoded strings.
        
    Returns
    -------
    analysis : dict
        Dictionary containing pattern analysis results.
    """
    if not sax_strings:
        return {'error': 'No SAX strings provided'}
    
    # Filter out empty strings
    valid_strings = [s for s in sax_strings if s]
    
    if not valid_strings:
        return {'error': 'No valid SAX strings found'}
    
    # Pattern frequency analysis
    pattern_freq = defaultdict(int)
    symbol_freq = defaultdict(int)
    
    lengths = []
    
    for sax_str in valid_strings:
        lengths.append(len(sax_str))
        pattern_freq[sax_str] += 1
        
        for symbol in sax_str:
            symbol_freq[symbol] += 1
    
    # Find common subsequences
    common_patterns = {}
    if valid_strings:
        min_len = min(lengths)
        for length in range(2, min(min_len + 1, 5)):  # Check patterns up to length 4
            pattern_count = defaultdict(int)
            for sax_str in valid_strings:
                for i in range(len(sax_str) - length + 1):
                    pattern = sax_str[i:i + length]
                    pattern_count[pattern] += 1
            
            # Keep patterns that appear in multiple strings
            common_patterns[length] = {
                pattern: count for pattern, count in pattern_count.items() 
                if count > 1
            }
    
    return {
        'total_strings': len(valid_strings),
        'avg_length': np.mean(lengths) if lengths else 0,
        'length_std': np.std(lengths) if lengths else 0,
        'unique_patterns': len(pattern_freq),
        'most_common_patterns': dict(sorted(pattern_freq.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]),
        'symbol_distribution': dict(symbol_freq),
        'common_subsequences': common_patterns
    }


def create_sax_vocabulary(sax_strings: List[str], min_frequency: int = 2) -> dict:
    """
    Create a vocabulary from SAX strings for further analysis.
    
    Parameters
    ----------
    sax_strings : list of str
        Collection of SAX strings.
    min_frequency : int, default=2
        Minimum frequency for a pattern to be included in vocabulary.
        
    Returns
    -------
    vocabulary : dict
        Dictionary mapping patterns to their frequencies.
    """
    pattern_freq = defaultdict(int)
    
    for sax_str in sax_strings:
        if sax_str:  # Skip empty strings
            pattern_freq[sax_str] += 1
    
    # Filter by minimum frequency
    vocabulary = {
        pattern: freq for pattern, freq in pattern_freq.items() 
        if freq >= min_frequency
    }
    
    return vocabulary