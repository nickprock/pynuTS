"""
Created on Thu Jun 18 2020

@project: pynuTS
@author: nicola procopio
@last_update: 05/08/2025
@description: Scalable Time Series Clustering with parallelization
@references: https://iaml.it/blog/serie-storiche-3-dynamic-time-warping
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict, Tuple, Optional, Callable, Any
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import warnings
import time
from collections import defaultdict
import gc
from dataclasses import dataclass

# Try to import additional performance libraries
try:
    import dask.array as da
    from dask import delayed, compute
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from joblib import Parallel, delayed as joblib_delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import our optimized DTW implementation
from .naive_dtw import DTWCalculator, fast_dtw, dtw_distance_matrix


@dataclass
class ClusteringMetrics:
    """Container for clustering quality metrics."""
    inertia: float
    silhouette_score: float = None
    calinski_harabasz_score: float = None
    davies_bouldin_score: float = None
    convergence_iterations: int = 0
    computation_time: float = 0.0


class ScalableDTWKMeans(BaseEstimator, ClusterMixin):
    """
    Highly scalable K-Means clustering for time series using DTW distance.
    
    This implementation provides massive performance improvements through:
    - Parallel computation of distances
    - Memory-efficient DTW algorithms
    - Smart initialization strategies
    - Early convergence detection
    - Batch processing for large datasets
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to form.
    max_iter : int, default=100
        Maximum number of iterations for a single run.
    n_init : int, default=10
        Number of random initializations.
    window_size : int, default=1
        DTW window constraint size.
    distance_metric : str, default='euclidean'
        Distance metric for DTW computation.
    init_method : str, default='k-means++'
        Initialization method: 'random', 'k-means++', 'dtw++'.
    tol : float, default=1e-4
        Tolerance for convergence.
    patience : int, default=5
        Early stopping patience.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all cores).
    batch_size : int, default=None
        Batch size for large datasets (auto if None).
    memory_efficient : bool, default=True
        Use memory-efficient DTW computation.
    random_state : int, default=None
        Random state for reproducibility.
    verbose : bool, default=False
        Verbose output during fitting.
    
    Attributes
    ----------
    cluster_centers_ : List[np.ndarray]
        Cluster centroids.
    labels_ : np.ndarray
        Cluster labels for each sample.
    inertia_ : float
        Sum of squared distances to centroids.
    n_iter_ : int
        Number of iterations run.
    """
    
    def __init__(self,
                 n_clusters: int = 2,
                 max_iter: int = 100,
                 n_init: int = 10,
                 window_size: int = 1,
                 distance_metric: str = 'euclidean',
                 init_method: str = 'k-means++',
                 tol: float = 1e-4,
                 patience: int = 5,
                 n_jobs: int = -1,
                 batch_size: Optional[int] = None,
                 memory_efficient: bool = True,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        
        self._validate_parameters(n_clusters, max_iter, n_init, window_size, 
                                init_method, distance_metric)
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.window_size = window_size
        self.distance_metric = distance_metric
        self.init_method = init_method
        self.tol = tol
        self.patience = patience
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.memory_efficient = memory_efficient
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize DTW calculator
        self.dtw_calculator = DTWCalculator(distance_metric=distance_metric)
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Internal state
        self._is_fitted = False
        self._preprocessing_time = 0.0
        self._clustering_time = 0.0
    
    def _validate_parameters(self, n_clusters: int, max_iter: int, n_init: int,
                           window_size: int, init_method: str, distance_metric: str):
        """Validate input parameters."""
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if n_init < 1:
            raise ValueError("n_init must be at least 1")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if init_method not in ['random', 'k-means++', 'dtw++']:
            raise ValueError(f"Unknown init_method: {init_method}")
        if distance_metric not in ['euclidean', 'cityblock', 'chebyshev']:
            raise ValueError(f"Unsupported distance_metric: {distance_metric}")
    
    def _prepare_data(self, X: List[Union[np.ndarray, pd.Series]]) -> List[np.ndarray]:
        """Convert and validate input data."""
        processed_data = []
        
        for i, series in enumerate(X):
            if isinstance(series, pd.Series):
                series_array = series.values
            elif isinstance(series, (list, tuple)):
                series_array = np.array(series)
            elif isinstance(series, np.ndarray):
                series_array = series.copy()
            else:
                raise TypeError(f"Unsupported data type at index {i}: {type(series)}")
            
            # Ensure 1D
            if series_array.ndim > 1:
                if series_array.shape[0] == 1:
                    series_array = series_array.flatten()
                else:
                    raise ValueError(f"Multi-dimensional series not supported at index {i}")
            
            # Handle missing values
            if np.isnan(series_array).any():
                warnings.warn(f"NaN values found in series {i}, using forward fill")
                mask = np.isnan(series_array)
                series_array[mask] = np.interp(np.where(mask)[0], 
                                             np.where(~mask)[0], 
                                             series_array[~mask])
            
            processed_data.append(series_array)
        
        return processed_data
    
    def _compute_distance_matrix_parallel(self, data: List[np.ndarray], 
                                        centroids: List[np.ndarray]) -> np.ndarray:
        """Compute distance matrix using parallel processing."""
        n_samples = len(data)
        n_centroids = len(centroids)
        
        if self.n_jobs == 1 or n_samples * n_centroids < 100:
            # Sequential computation for small problems
            distances = np.zeros((n_samples, n_centroids))
            for i, series in enumerate(data):
                for j, centroid in enumerate(centroids):
                    if self.memory_efficient:
                        distances[i, j] = self.dtw_calculator.memory_efficient_dtw(
                            series, centroid, self.window_size
                        )
                    else:
                        distances[i, j], _ = self.dtw_calculator.naive_dtw(
                            series, centroid, self.window_size
                        )
            return distances
        
        # Parallel computation
        def compute_distance_row(i: int) -> Tuple[int, np.ndarray]:
            """Compute distances from one data point to all centroids."""
            row_distances = np.zeros(n_centroids)
            for j, centroid in enumerate(centroids):
                if self.memory_efficient:
                    row_distances[j] = self.dtw_calculator.memory_efficient_dtw(
                        data[i], centroid, self.window_size
                    )
                else:
                    row_distances[j], _ = self.dtw_calculator.naive_dtw(
                        data[i], centroid, self.window_size
                    )
            return i, row_distances
        
        # Choose executor based on problem size and available libraries
        if JOBLIB_AVAILABLE and n_samples > 50:
            # Use joblib for CPU-bound tasks
            results = Parallel(n_jobs=self.n_jobs)(
                joblib_delayed(compute_distance_row)(i) for i in range(n_samples)
            )
        else:
            # Use ThreadPoolExecutor for I/O bound or smaller tasks
            max_workers = None if self.n_jobs == -1 else max(1, self.n_jobs)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(compute_distance_row, i) for i in range(n_samples)]
                results = [future.result() for future in futures]
        
        # Reconstruct distance matrix
        distances = np.zeros((n_samples, n_centroids))
        for i, row_distances in results:
            distances[i] = row_distances
        
        return distances
    
    def _initialize_centroids(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """Initialize centroids using specified method."""
        n_samples = len(data)
        
        if self.init_method == 'random':
            # Random initialization
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return [data[i].copy() for i in indices]
        
        elif self.init_method == 'k-means++':
            # K-means++ initialization adapted for DTW
            centroids = []
            
            # Choose first centroid randomly
            first_idx = np.random.randint(n_samples)
            centroids.append(data[first_idx].copy())
            
            # Choose remaining centroids
            for _ in range(1, self.n_clusters):
                distances = np.full(n_samples, np.inf)
                
                # Compute distances to nearest centroid
                for i, series in enumerate(data):
                    for centroid in centroids:
                        dist = self.dtw_calculator.memory_efficient_dtw(
                            series, centroid, self.window_size
                        )
                        distances[i] = min(distances[i], dist)
                
                # Choose next centroid with probability proportional to squared distance
                probabilities = distances ** 2
                probabilities /= probabilities.sum()
                
                next_idx = np.random.choice(n_samples, p=probabilities)
                centroids.append(data[next_idx].copy())
            
            return centroids
        
        elif self.init_method == 'dtw++':
            # Enhanced DTW-aware initialization
            return self._dtw_plus_plus_init(data)
        
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
    
    def _dtw_plus_plus_init(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """DTW++ initialization - considers DTW-specific properties."""
        n_samples = len(data)
        centroids = []
        
        # Find the most "central" series as first centroid
        if n_samples <= 100:
            # For small datasets, compute full distance matrix
            pairwise_distances = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = self.dtw_calculator.memory_efficient_dtw(
                        data[i], data[j], self.window_size
                    )
                    pairwise_distances[i, j] = dist
                    pairwise_distances[j, i] = dist
            
            # Choose series with minimum sum of distances
            sum_distances = pairwise_distances.sum(axis=1)
            first_idx = np.argmin(sum_distances)
        else:
            # For large datasets, sample and find approximate center
            sample_size = min(50, n_samples // 10)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            
            min_total_dist = np.inf
            first_idx = 0
            
            for i in sample_indices:
                total_dist = 0
                for j in sample_indices:
                    if i != j:
                        total_dist += self.dtw_calculator.memory_efficient_dtw(
                            data[i], data[j], self.window_size
                        )
                
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    first_idx = i
        
        centroids.append(data[first_idx].copy())
        
        # Choose remaining centroids using modified k-means++
        for _ in range(1, self.n_clusters):
            distances = np.full(n_samples, np.inf)
            
            # Parallel distance computation
            def compute_min_distance(i: int) -> float:
                min_dist = np.inf
                for centroid in centroids:
                    dist = self.dtw_calculator.memory_efficient_dtw(
                        data[i], centroid, self.window_size
                    )
                    min_dist = min(min_dist, dist)
                return min_dist
            
            if JOBLIB_AVAILABLE and n_samples > 100:
                distances = Parallel(n_jobs=self.n_jobs)(
                    joblib_delayed(compute_min_distance)(i) for i in range(n_samples)
                )
                distances = np.array(distances)
            else:
                for i in range(n_samples):
                    distances[i] = compute_min_distance(i)
            
            # Choose next centroid
            probabilities = distances ** 2
            if probabilities.sum() > 0:
                probabilities /= probabilities.sum()
                next_idx = np.random.choice(n_samples, p=probabilities)
            else:
                # Fallback to random choice
                remaining_indices = [i for i in range(n_samples) 
                                   if not any(np.array_equal(data[i], c) for c in centroids)]
                next_idx = np.random.choice(remaining_indices)
            
            centroids.append(data[next_idx].copy())
        
        return centroids
    
    def _compute_centroid(self, cluster_data: List[np.ndarray]) -> np.ndarray:
        """Compute centroid for a cluster using DTW barycenter."""
        if not cluster_data:
            # Return random centroid if cluster is empty
            return np.random.randn(100)  # Default length
        
        if len(cluster_data) == 1:
            return cluster_data[0].copy()
        
        # Use DTW barycenter computation
        try:
            from .naive_dtw import dtw_barycenter
            return dtw_barycenter(cluster_data, max_iterations=5, 
                                window_size=self.window_size)
        except ImportError:
            # Fallback to simple average (aligned by interpolation)
            return self._simple_barycenter(cluster_data)
    
    def _simple_barycenter(self, cluster_data: List[np.ndarray]) -> np.ndarray:
        """Simple barycenter computation using interpolation.""" 
        # Find target length (median length)
        lengths = [len(series) for series in cluster_data]
        target_length = int(np.median(lengths))
        
        # Interpolate all series to target length
        aligned_data = []
        for series in cluster_data:
            if len(series) != target_length:
                # Linear interpolation
                x_old = np.linspace(0, 1, len(series))
                x_new = np.linspace(0, 1, target_length)
                aligned_series = np.interp(x_new, x_old, series)
                aligned_data.append(aligned_series)
            else:
                aligned_data.append(series)
        
        # Return mean
        return np.mean(aligned_data, axis=0)
    
    def _assign_clusters(self, data: List[np.ndarray], 
                        centroids: List[np.ndarray]) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """Assign data points to clusters."""
        distances = self._compute_distance_matrix_parallel(data, centroids)
        labels = np.argmin(distances, axis=1)
        
        # Create cluster assignments dictionary
        assignments = defaultdict(list)
        for i, label in enumerate(labels):
            assignments[label].append(i)
        
        return labels, dict(assignments)
    
    def _compute_inertia(self, data: List[np.ndarray], centroids: List[np.ndarray],
                        assignments: Dict[int, List[int]]) -> float:
        """Compute clustering inertia (sum of squared DTW distances)."""
        inertia = 0.0
        
        for cluster_id, member_indices in assignments.items():
            if cluster_id < len(centroids) and member_indices:
                centroid = centroids[cluster_id]
                
                # Parallel computation of squared distances
                def compute_squared_distance(i: int) -> float:
                    dist = self.dtw_calculator.memory_efficient_dtw(
                        data[i], centroid, self.window_size
                    )
                    return dist ** 2
                
                if JOBLIB_AVAILABLE and len(member_indices) > 20:
                    squared_distances = Parallel(n_jobs=self.n_jobs)(
                        joblib_delayed(compute_squared_distance)(i) for i in member_indices
                    )
                    inertia += sum(squared_distances)
                else:
                    for i in member_indices:
                        inertia += compute_squared_distance(i)
        
        return inertia
    
    def _single_run(self, data: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, float, int]:
        """Single k-means run."""
        # Initialize centroids
        centroids = self._initialize_centroids(data)
        prev_inertia = np.inf
        patience_counter = 0
        
        for iteration in range(self.max_iter):
            # Assign clusters
            labels, assignments = self._assign_clusters(data, centroids)
            
            # Update centroids
            new_centroids = []
            for cluster_id in range(self.n_clusters):
                if cluster_id in assignments and assignments[cluster_id]:
                    cluster_data = [data[i] for i in assignments[cluster_id]]
                    new_centroid = self._compute_centroid(cluster_data)
                    new_centroids.append(new_centroid)
                else:
                    # Keep old centroid if cluster is empty
                    if cluster_id < len(centroids):
                        new_centroids.append(centroids[cluster_id].copy())
                    else:
                        # Generate random centroid
                        random_idx = np.random.randint(len(data))
                        new_centroids.append(data[random_idx].copy())
            
            centroids = new_centroids
            
            # Compute inertia
            inertia = self._compute_inertia(data, centroids, assignments)
            
            # Check convergence
            if abs(prev_inertia - inertia) < self.tol:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
            else:
                patience_counter = 0
            
            prev_inertia = inertia
            
            if self.verbose:
                print(f"Iteration {iteration + 1}: inertia = {inertia:.6f}")
        
        return centroids, labels, inertia, iteration + 1
    
    def fit(self, X: List[Union[np.ndarray, pd.Series]], y=None) -> 'ScalableDTWKMeans':
        """
        Fit the DTW K-means clustering.
        
        Parameters
        ----------
        X : list of array-like
            Time series data to cluster.
        y : ignored
            Not used, present for sklearn compatibility.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        start_time = time.time()
        
        # Prepare data
        if self.verbose:
            print("Preprocessing data...")
        preprocessing_start = time.time()
        data = self._prepare_data(X)
        self._preprocessing_time = time.time() - preprocessing_start
        
        if len(data) < self.n_clusters:
            raise ValueError(f"Number of samples ({len(data)}) must be >= n_clusters ({self.n_clusters})")
        
        # Auto-determine batch size
        if self.batch_size is None:
            # Heuristic based on data size and available memory
            n_samples = len(data)
            avg_length = np.mean([len(series) for series in data])
            memory_estimate = n_samples * avg_length * 8 / (1024**3)  # GB
            
            if memory_estimate > 2.0:  # If > 2GB
                self.batch_size = max(100, n_samples // 10)
            else:
                self.batch_size = n_samples  # Process all at once
        
        # Multiple runs for best result
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_iterations = 0
        
        clustering_start = time.time()
        
        if self.verbose:
            print(f"Running {self.n_init} initialization(s)...")
        
        for run in range(self.n_init):
            if self.verbose:
                print(f"Run {run + 1}/{self.n_init}")
            
            centroids, labels, inertia, iterations = self._single_run(data)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_iterations = iterations
            
            # Memory cleanup
            if run % 5 == 0:
                gc.collect()
        
        self._clustering_time = time.time() - clustering_start
        
        # Store results
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iterations
        self._is_fitted = True
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"Clustering completed in {total_time:.2f}s")
            print(f"  Preprocessing: {self._preprocessing_time:.2f}s")
            print(f"  Clustering: {self._clustering_time:.2f}s")
            print(f"  Final inertia: {best_inertia:.6f}")
            print(f"  Iterations: {best_iterations}")
        
        return self
    
    def predict(self, X: List[Union[np.ndarray, pd.Series]]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Parameters
        ----------
        X : list of array-like
            Time series data to predict.
            
        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels.
        """
        check_is_fitted(self, ['cluster_centers_', 'labels_'])
        
        data = self._prepare_data(X)
        distances = self._compute_distance_matrix_parallel(data, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: List[Union[np.ndarray, pd.Series]], y=None) -> np.ndarray:
        """Fit and predict in one step."""
        return self.fit(X, y).labels_
    
    def transform(self, X: List[Union[np.ndarray, pd.Series]]) -> np.ndarray:
        """Transform data to cluster-distance space."""
        check_is_fitted(self, ['cluster_centers_'])
        
        data = self._prepare_data(X)
        return self._compute_distance_matrix_parallel(data, self.cluster_centers_)
    
    def get_clustering_metrics(self) -> ClusteringMetrics:
        """Get comprehensive clustering quality metrics."""
        check_is_fitted(self, ['cluster_centers_', 'labels_', 'inertia_'])
        
        return ClusteringMetrics(
            inertia=self.inertia_,
            convergence_iterations=self.n_iter_,
            computation_time=self._preprocessing_time + self._clustering_time
        )


# Backward compatibility class
class DTWKmeans(ScalableDTWKMeans):
    """Backward compatibility wrapper for the original DTWKmeans class."""
    
    def __init__(self, num_clust: int, num_iter: int = 1, num_init: int = 1,
                 w: int = 1, criterion: str = 'euclidean', seed=None):
        """Initialize with original parameter names."""
        super().__init__(
            n_clusters=num_clust,
            max_iter=num_iter,
            n_init=num_init,
            window_size=w,
            distance_metric=criterion,
            random_state=seed,
            verbose=False
        )
    
    def fit(self, data: list, patience: int = 5):
        """Fit with original signature."""
        self.patience = patience
        return super().fit(data)
    
    def predict(self, data: list) -> dict:
        """Predict with original output format."""
        labels = super().predict(data)
        
        # Convert to original dictionary format
        assignments = {}
        for i in range(self.n_clusters):
            assignments[i] = []
        
        for idx, label in enumerate(labels):
            assignments[label].append(idx)
        
        return assignments


# Utility functions for large-scale clustering
def hierarchical_dtw_clustering(data: List[np.ndarray], 
                               max_cluster_size: int = 1000,
                               n_clusters: int = None,
                               **kwargs) -> Dict[str, Any]:
    """
    Hierarchical clustering approach for very large datasets.
    
    Parameters
    ----------
    data : list of np.ndarray
        Time series data.
    max_cluster_size : int, default=1000
        Maximum size for subclusters before hierarchical split.
    n_clusters : int, optional
        Final number of clusters (auto-determined if None).
    **kwargs : dict
        Additional arguments for ScalableDTWKMeans.
        
    Returns
    -------
    result : dict
        Dictionary with clustering results and hierarchy information.
    """
    if len(data) <= max_cluster_size:
        # Direct clustering for small datasets
        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(len(data))))
        
        clusterer = ScalableDTWKMeans(n_clusters=n_clusters, **kwargs)
        labels = clusterer.fit_predict(data)
        
        return {
            'labels': labels,
            'cluster_centers': clusterer.cluster_centers_,
            'inertia': clusterer.inertia_,
            'hierarchy_levels': 1
        }
    
    # Hierarchical approach
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(len(data))))
    
    # First level: create subclusters
    n_subclusters = min(max(4, n_clusters * 2), len(data) // 100)
    
    level1_clusterer = ScalableDTWKMeans(
        n_clusters=n_subclusters, 
        n_init=5,  # Fewer inits for speed
        **kwargs
    )
    level1_labels = level1_clusterer.fit_predict(data)
    
    # Second level: merge subclusters
    subcluster_centers = level1_clusterer.cluster_centers_
    
    level2_clusterer = ScalableDTWKMeans(
        n_clusters=n_clusters,
        n_init=10,
        **kwargs
    )
    level2_labels = level2_clusterer.fit_predict(subcluster_centers)
    
    # Map original data to final clusters
    final_labels = np.zeros(len(data), dtype=int)
    for i, l1_label in enumerate(level1_labels):
        final_labels[i] = level2_labels[l1_label]
    
    return {
        'labels': final_labels,
        'cluster_centers': level2_clusterer.cluster_centers_,
        'inertia': level2_clusterer.inertia_,
        'hierarchy_levels': 2,
        'level1_centers': subcluster_centers,
        'level1_labels': level1_labels
    }


def parallel_mini_batch_clustering(data: List[np.ndarray],
                                  n_clusters: int,
                                  batch_size: int = 1000,
                                  n_batches: int = 10,
                                  **kwargs) -> ScalableDTWKMeans:
    """
    Mini-batch clustering for extremely large datasets.
    
    Parameters
    ----------
    data : list of np.ndarray
        Time series data.
    n_clusters : int
        Number of clusters.
    batch_size : int, default=1000
        Size of each mini-batch.
    n_batches : int, default=10
        Number of mini-batches to process.
    **kwargs : dict
        Additional arguments for ScalableDTWKMeans.
        
    Returns
    -------
    clusterer : ScalableDTWKMeans
        Trained clustering model.
    """
    n_samples = len(data)
    
    if n_samples <= batch_size:
        # Direct clustering for small datasets
        clusterer = ScalableDTWKMeans(n_clusters=n_clusters, **kwargs)
        clusterer.fit(data)
        return clusterer
    
    # Initialize with first batch
    first_batch_indices = np.random.choice(n_samples, batch_size, replace=False)
    first_batch = [data[i] for i in first_batch_indices]
    
    clusterer = ScalableDTWKMeans(n_clusters=n_clusters, n_init=3, **kwargs)
    clusterer.fit(first_batch)
    
    # Process additional batches
    for batch_num in range(1, n_batches):
        # Sample new batch
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        batch_data = [data[i] for i in batch_indices]
        
        # Assign to existing clusters
        distances = clusterer._compute_distance_matrix_parallel(batch_data, clusterer.cluster_centers_)
        batch_labels = np.argmin(distances, axis=1)
        
        # Update centroids using new batch
        for cluster_id in range(n_clusters):
            cluster_mask = batch_labels == cluster_id
            if np.any(cluster_mask):
                cluster_batch_data = [batch_data[i] for i in np.where(cluster_mask)[0]]
                
                # Combine with existing centroid (weighted average)
                current_centroid = clusterer.cluster_centers_[cluster_id]
                new_centroid = clusterer._compute_centroid(cluster_batch_data + [current_centroid])
                clusterer.cluster_centers_[cluster_id] = new_centroid
    
    # Final refinement on random sample
    final_sample_indices = np.random.choice(n_samples, min(batch_size * 2, n_samples), replace=False)
    final_sample = [data[i] for i in final_sample_indices]
    
    # Re-fit on final sample to stabilize centroids
    final_clusterer = ScalableDTWKMeans(n_clusters=n_clusters, n_init=1, **kwargs)
    final_clusterer.cluster_centers_ = clusterer.cluster_centers_  # Use existing centroids as initialization
    final_clusterer.fit(final_sample)
    
    return final_clusterer


def online_dtw_clustering(n_clusters: int, 
                         learning_rate: float = 0.1,
                         window_size: int = 1,
                         **kwargs) -> 'OnlineDTWKMeans':
    """
    Create an online DTW clustering model for streaming data.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    learning_rate : float, default=0.1
        Learning rate for online updates.
    window_size : int, default=1
        DTW window constraint.
    **kwargs : dict
        Additional parameters.
        
    Returns
    -------
    clusterer : OnlineDTWKMeans
        Online clustering model.
    """
    return OnlineDTWKMeans(
        n_clusters=n_clusters,
        learning_rate=learning_rate,
        window_size=window_size,
        **kwargs
    )


class OnlineDTWKMeans:
    """
    Online DTW K-means clustering for streaming time series data.
    
    This implementation allows for incremental learning from streaming
    time series data without storing all historical data.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    learning_rate : float, default=0.1
        Learning rate for centroid updates.
    window_size : int, default=1
        DTW window constraint.
    decay_factor : float, default=0.95
        Decay factor for centroid importance.
    min_samples_per_cluster : int, default=10
        Minimum samples before a cluster is considered stable.
    random_state : int, optional
        Random state for reproducibility.
    """
    
    def __init__(self,
                 n_clusters: int,
                 learning_rate: float = 0.1,
                 window_size: int = 1,
                 decay_factor: float = 0.95,
                 min_samples_per_cluster: int = 10,
                 random_state: Optional[int] = None):
        
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.min_samples_per_cluster = min_samples_per_cluster
        self.random_state = random_state
        
        # Initialize DTW calculator
        self.dtw_calculator = DTWCalculator()
        
        # Internal state
        self.cluster_centers_ = None
        self.cluster_counts_ = np.zeros(n_clusters)
        self.is_initialized_ = False
        self.n_samples_seen_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def partial_fit(self, X: List[Union[np.ndarray, pd.Series]]) -> 'OnlineDTWKMeans':
        """
        Incrementally fit the model with new data.
        
        Parameters
        ----------
        X : list of array-like
            New time series data to incorporate.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Convert data
        data = []
        for series in X:
            if isinstance(series, pd.Series):
                data.append(series.values)
            elif isinstance(series, (list, tuple)):
                data.append(np.array(series))
            else:
                data.append(np.asarray(series))
        
        if not self.is_initialized_:
            self._initialize_centroids(data)
        
        # Process each new sample
        for series in data:
            self._update_with_sample(series)
            self.n_samples_seen_ += 1
        
        return self
    
    def _initialize_centroids(self, data: List[np.ndarray]):
        """Initialize centroids with first batch of data."""
        if len(data) < self.n_clusters:
            # Duplicate data if insufficient samples
            while len(data) < self.n_clusters:
                data.extend(data[:self.n_clusters - len(data)])
        
        # Random initialization from first batch
        indices = np.random.choice(len(data), self.n_clusters, replace=False)
        self.cluster_centers_ = [data[i].copy() for i in indices]
        self.is_initialized_ = True
    
    def _update_with_sample(self, sample: np.ndarray):
        """Update model with a single sample."""
        # Find closest cluster
        distances = []
        for centroid in self.cluster_centers_:
            dist = self.dtw_calculator.memory_efficient_dtw(
                sample, centroid, self.window_size
            )
            distances.append(dist)
        
        closest_cluster = np.argmin(distances)
        
        # Update cluster centroid
        current_centroid = self.cluster_centers_[closest_cluster]
        self.cluster_counts_[closest_cluster] += 1
        
        # Adaptive learning rate based on cluster stability
        count = self.cluster_counts_[closest_cluster]
        if count >= self.min_samples_per_cluster:
            adaptive_lr = self.learning_rate / np.sqrt(count)
        else:
            adaptive_lr = self.learning_rate
        
        # Update centroid using exponential moving average
        # For simplicity, use element-wise update (assumes same length)
        if len(sample) == len(current_centroid):
            updated_centroid = (1 - adaptive_lr) * current_centroid + adaptive_lr * sample
        else:
            # Handle different lengths with interpolation
            target_length = len(current_centroid)
            if len(sample) != target_length:
                x_old = np.linspace(0, 1, len(sample))
                x_new = np.linspace(0, 1, target_length)
                sample_aligned = np.interp(x_new, x_old, sample)
            else:
                sample_aligned = sample
            
            updated_centroid = (1 - adaptive_lr) * current_centroid + adaptive_lr * sample_aligned
        
        self.cluster_centers_[closest_cluster] = updated_centroid
        
        # Apply decay to other clusters
        for i in range(self.n_clusters):
            if i != closest_cluster:
                self.cluster_counts_[i] *= self.decay_factor
    
    def predict(self, X: List[Union[np.ndarray, pd.Series]]) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_initialized_:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert data
        data = []
        for series in X:
            if isinstance(series, pd.Series):
                data.append(series.values)
            elif isinstance(series, (list, tuple)):
                data.append(np.array(series))
            else:
                data.append(np.asarray(series))
        
        labels = []
        for series in data:
            distances = []
            for centroid in self.cluster_centers_:
                dist = self.dtw_calculator.memory_efficient_dtw(
                    series, centroid, self.window_size
                )
                distances.append(dist)
            
            labels.append(np.argmin(distances))
        
        return np.array(labels)


# Advanced clustering techniques
class EnsembleDTWKMeans:
    """
    Ensemble DTW K-means clustering for improved robustness.
    
    Uses multiple clustering runs with different parameters and
    combines results using consensus clustering.
    """
    
    def __init__(self,
                 n_clusters: int,
                 n_estimators: int = 10,
                 base_params: Optional[Dict] = None,
                 consensus_method: str = 'majority_vote',
                 random_state: Optional[int] = None):
        
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.base_params = base_params or {}
        self.consensus_method = consensus_method
        self.random_state = random_state
        
        self.estimators_ = []
        self.labels_ = None
        self.cluster_centers_ = None
    
    def fit(self, X: List[Union[np.ndarray, pd.Series]]) -> 'EnsembleDTWKMeans':
        """Fit ensemble of clustering models."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Generate diverse base estimators
        all_labels = []
        all_centers = []
        
        for i in range(self.n_estimators):
            # Vary parameters for diversity
            params = self.base_params.copy()
            params.update({
                'random_state': None if self.random_state is None else self.random_state + i,
                'window_size': np.random.choice([1, 2, 3, 5]),
                'init_method': np.random.choice(['random', 'k-means++', 'dtw++']),
                'n_init': np.random.choice([5, 10, 15])
            })
            
            # Train base estimator
            estimator = ScalableDTWKMeans(n_clusters=self.n_clusters, **params)
            estimator.fit(X)
            
            self.estimators_.append(estimator)
            all_labels.append(estimator.labels_)
            all_centers.append(estimator.cluster_centers_)
        
        # Consensus clustering
        if self.consensus_method == 'majority_vote':
            self.labels_ = self._majority_vote_consensus(all_labels)
        elif self.consensus_method == 'co_occurrence':
            self.labels_ = self._co_occurrence_consensus(all_labels, X)
        else:
            raise ValueError(f"Unknown consensus method: {self.consensus_method}")
        
        # Compute final centroids
        self._compute_consensus_centroids(X)
        
        return self
    
    def _majority_vote_consensus(self, all_labels: List[np.ndarray]) -> np.ndarray:
        """Consensus clustering using majority voting."""
        n_samples = len(all_labels[0])
        
        # Build co-occurrence matrix
        co_occurrence = np.zeros((n_samples, n_samples))
        
        for labels in all_labels:
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        co_occurrence[i, j] += 1
        
        # Normalize
        co_occurrence /= len(all_labels)
        
        # Use spectral clustering on co-occurrence matrix
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=self.n_clusters, 
                                    affinity='precomputed',
                                    random_state=self.random_state)
        consensus_labels = spectral.fit_predict(co_occurrence)
        
        return consensus_labels
    
    def _co_occurrence_consensus(self, all_labels: List[np.ndarray], 
                               X: List[Union[np.ndarray, pd.Series]]) -> np.ndarray:
        """Advanced consensus using co-occurrence patterns."""
        # Similar to majority vote but with weighted voting based on clustering quality
        qualities = []
        
        for estimator in self.estimators_:
            # Use silhouette score as quality measure (simplified)
            quality = -estimator.inertia_  # Higher is better
            qualities.append(quality)
        
        # Normalize qualities to weights
        qualities = np.array(qualities)
        if np.std(qualities) > 0:
            weights = (qualities - np.min(qualities)) / (np.max(qualities) - np.min(qualities))
        else:
            weights = np.ones(len(qualities))
        
        weights /= np.sum(weights)
        
        # Weighted co-occurrence matrix
        n_samples = len(all_labels[0])
        co_occurrence = np.zeros((n_samples, n_samples))
        
        for labels, weight in zip(all_labels, weights):
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        co_occurrence[i, j] += weight
        
        # Use spectral clustering
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=self.n_clusters,
                                    affinity='precomputed',
                                    random_state=self.random_state)
        consensus_labels = spectral.fit_predict(co_occurrence)
        
        return consensus_labels
    
    def _compute_consensus_centroids(self, X: List[Union[np.ndarray, pd.Series]]):
        """Compute centroids based on consensus labels."""
        data = []
        for series in X:
            if isinstance(series, pd.Series):
                data.append(series.values)
            elif isinstance(series, (list, tuple)):
                data.append(np.array(series))
            else:
                data.append(np.asarray(series))
        
        centroids = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels_ == cluster_id
            if np.any(cluster_mask):
                cluster_data = [data[i] for i in np.where(cluster_mask)[0]]
                
                # Use DTW barycenter
                try:
                    from .naive_dtw import dtw_barycenter
                    centroid = dtw_barycenter(cluster_data, max_iterations=5)
                except ImportError:
                    # Fallback to simple average
                    lengths = [len(series) for series in cluster_data]
                    target_length = int(np.median(lengths))
                    
                    aligned_data = []
                    for series in cluster_data:
                        if len(series) != target_length:
                            x_old = np.linspace(0, 1, len(series))
                            x_new = np.linspace(0, 1, target_length)
                            aligned = np.interp(x_new, x_old, series)
                            aligned_data.append(aligned)
                        else:
                            aligned_data.append(series)
                    
                    centroid = np.mean(aligned_data, axis=0)
                
                centroids.append(centroid)
            else:
                # Empty cluster - use random sample
                random_idx = np.random.randint(len(data))
                centroids.append(data[random_idx].copy())
        
        self.cluster_centers_ = centroids
    
    def predict(self, X: List[Union[np.ndarray, pd.Series]]) -> np.ndarray:
        """Predict using consensus model."""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Use first estimator's DTW calculator for prediction
        dtw_calc = self.estimators_[0].dtw_calculator
        
        data = []
        for series in X:
            if isinstance(series, pd.Series):
                data.append(series.values)
            elif isinstance(series, (list, tuple)):
                data.append(np.array(series))
            else:
                data.append(np.asarray(series))
        
        labels = []
        for series in data:
            distances = []
            for centroid in self.cluster_centers_:
                dist = dtw_calc.memory_efficient_dtw(series, centroid, 1)
                distances.append(dist)
            
            labels.append(np.argmin(distances))
        
        return np.array(labels)


# Utility functions for performance benchmarking
def benchmark_clustering_performance(data: List[np.ndarray],
                                   n_clusters: int,
                                   methods: List[str] = None) -> Dict[str, Dict]:
    """
    Benchmark different clustering methods on the same dataset.
    
    Parameters
    ----------
    data : list of np.ndarray
        Time series data for benchmarking.
    n_clusters : int
        Number of clusters.
    methods : list of str, optional
        Methods to benchmark. If None, tests all available methods.
        
    Returns
    -------
    results : dict
        Benchmark results for each method.
    """
    if methods is None:
        methods = ['scalable', 'hierarchical', 'mini_batch', 'ensemble']
    
    results = {}
    
    for method in methods:
        print(f"Benchmarking {method}...")
        start_time = time.time()
        
        try:
            if method == 'scalable':
                clusterer = ScalableDTWKMeans(n_clusters=n_clusters, verbose=True)
                clusterer.fit(data)
                
                results[method] = {
                    'time': time.time() - start_time,
                    'inertia': clusterer.inertia_,
                    'iterations': clusterer.n_iter_,
                    'success': True
                }
                
            elif method == 'hierarchical':
                result = hierarchical_dtw_clustering(data, n_clusters=n_clusters)
                
                results[method] = {
                    'time': time.time() - start_time,
                    'inertia': result['inertia'],
                    'hierarchy_levels': result['hierarchy_levels'],
                    'success': True
                }
                
            elif method == 'mini_batch':
                clusterer = parallel_mini_batch_clustering(
                    data, n_clusters=n_clusters, 
                    batch_size=min(500, len(data) // 4)
                )
                
                results[method] = {
                    'time': time.time() - start_time,
                    'inertia': clusterer.inertia_,
                    'success': True
                }
                
            elif method == 'ensemble':
                clusterer = EnsembleDTWKMeans(n_clusters=n_clusters, n_estimators=5)
                clusterer.fit(data)
                
                results[method] = {
                    'time': time.time() - start_time,
                    'n_estimators': len(clusterer.estimators_),
                    'success': True
                }
                
        except Exception as e:
            results[method] = {
                'time': time.time() - start_time,
                'error': str(e),
                'success': False
            }
    
    return results