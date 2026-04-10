import warnings
from typing import Optional, Union, Tuple

import numpy as np
from numba import jit, prange
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted


# =====================================================================
# Numba-accelerated core functions for performance
# =====================================================================


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _wasserstein_distance_1d_batch_numba(mu, nu, p):
    """
    Compute 1D Wasserstein distances between batches using Numba JIT.

    Parameters
    ----------
    mu : ndarray of shape (..., n_atoms)
        First batch of sorted 1D measures.
    nu : ndarray of shape (..., n_atoms)
        Second batch of sorted 1D measures.
    p : int
        Order of Wasserstein distance (1 or 2).

    Returns
    -------
    distances : ndarray of shape (...)
        Wasserstein distances.
    """
    shape = mu.shape
    n_elements = 1
    for i in range(len(shape) - 1):
        n_elements *= shape[i]

    n_atoms = shape[-1]
    distances = np.empty(n_elements, dtype=np.float64)

    # Flatten to 2D for easier indexing
    mu_flat = mu.reshape(n_elements, n_atoms)
    nu_flat = nu.reshape(n_elements, n_atoms)

    if p == 1:
        for i in prange(n_elements):
            dist = 0.0
            for j in range(n_atoms):
                dist += abs(mu_flat[i, j] - nu_flat[i, j])
            distances[i] = dist / n_atoms
    else:  # p == 2
        for i in prange(n_elements):
            dist = 0.0
            for j in range(n_atoms):
                diff = mu_flat[i, j] - nu_flat[i, j]
                dist += diff * diff
            distances[i] = np.sqrt(dist / n_atoms)

    # Reshape back to original shape (minus last dimension)
    return distances.reshape(shape[:-1])


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_distances_chunked_numba(seq_proj_chunk, cent_proj, p, n_projections):
    """
    Compute distance matrix for a chunk of sequences using Numba JIT.

    Parameters
    ----------
    seq_proj_chunk : ndarray of shape (chunk_size, n_projections, window_size)
        Projected sequences chunk.
    cent_proj : ndarray of shape (n_clusters, n_projections, window_size)
        Projected centroids.
    p : int
        Order of Wasserstein distance.
    n_projections : int
        Number of projections.

    Returns
    -------
    distances : ndarray of shape (chunk_size, n_clusters)
        Distances from chunk sequences to centroids.
    """
    chunk_size = seq_proj_chunk.shape[0]
    n_clusters = cent_proj.shape[0]
    window_size = seq_proj_chunk.shape[2]

    distances = np.empty((chunk_size, n_clusters), dtype=np.float64)

    for i in prange(chunk_size):
        for j in range(n_clusters):
            total_dist = 0.0

            for k in range(n_projections):
                # Compute 1D Wasserstein distance for this projection
                proj_dist = 0.0
                if p == 1:
                    for t in range(window_size):
                        proj_dist += abs(seq_proj_chunk[i, k, t] - cent_proj[j, k, t])
                    proj_dist /= window_size
                else:  # p == 2
                    for t in range(window_size):
                        diff = seq_proj_chunk[i, k, t] - cent_proj[j, k, t]
                        proj_dist += diff * diff
                    proj_dist = np.sqrt(proj_dist / window_size)

                total_dist += proj_dist

            distances[i, j] = total_dist / n_projections

    return distances


@jit(nopython=True, cache=True)
def _wasserstein_barycenter_1d_numba(measures, p):
    """
    Compute 1D Wasserstein barycenter using Numba JIT.

    Parameters
    ----------
    measures : ndarray of shape (n_measures, n_atoms)
        Sorted 1D measures.
    p : int
        Order of Wasserstein distance (1 or 2).

    Returns
    -------
    barycenter : ndarray of shape (n_atoms,)
        Wasserstein barycenter.
    """
    n_measures = measures.shape[0]
    n_atoms = measures.shape[1]
    barycenter = np.empty(n_atoms, dtype=np.float64)

    if p == 1:
        # Median for W1
        for j in range(n_atoms):
            # Sort values for this atom across measures
            values = np.empty(n_measures, dtype=np.float64)
            for i in range(n_measures):
                values[i] = measures[i, j]
            values.sort()

            # Compute median
            if n_measures % 2 == 0:
                barycenter[j] = (
                    values[n_measures // 2 - 1] + values[n_measures // 2]
                ) / 2.0
            else:
                barycenter[j] = values[n_measures // 2]
    else:  # p == 2
        # Mean for W2
        for j in range(n_atoms):
            total = 0.0
            for i in range(n_measures):
                total += measures[i, j]
            barycenter[j] = total / n_measures

    return barycenter


class SlicedWassersteinKMeans(BaseEstimator, ClusterMixin):
    """
    Sliced Wasserstein k-means clustering for multidimensional time series.

    This implementation extends Wasserstein k-means to multidimensional data by using
    sliced Wasserstein distances computed via 1D projections.

    Paper: https://www.aimspress.com/article/doi/10.3934/DSFE.2025016

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form. Must be >= 2.

    window_size : int, default=35
        Size of the sliding window (h1 parameter in the paper).
        Must be > lifting_size.

    lifting_size : int, default=7
        Sliding window offset parameter (h2 parameter in the paper).
        Smaller values increase overlap between sequences. Must be > 0.

    n_projections : int, default=9
        Number of random projection directions for sliced Wasserstein distance.
        More projections give better approximation but slower computation.

    p : int, default=1
        Parameter for p-Wasserstein distance. Must be 1 or 2.
        - p=1: Uses L1 distance (more robust to outliers)
        - p=2: Uses L2 distance (smoother gradients)

    max_iter : int, default=100
        Maximum number of iterations for the k-means algorithm.

    tol : float, default=1e-6
        Relative tolerance for convergence criterion.

    random_state : int, RandomState instance or None, default=None
        Controls random number generation for centroid initialization and projections.

    n_init : int, default=10
        Number of random initializations to try. Best result is kept.
        Reduce this value (e.g., 3) for faster training on large datasets.

    standardize : bool, default=False
        Whether to standardize returns coordinate-wise before clustering.

    temperature : float, default=1.0
        Temperature parameter for probability computation via softmax.
        Lower values make probabilities more peaked.

    chunk_size : int, default=1000
        Number of sequences to process in each chunk for distance computation.
        Larger values are faster but use more memory. Smaller values reduce
        memory usage but may be slower. Optimal value depends on dataset size.

    use_minibatch : bool, default=False
        Whether to use mini-batch K-means instead of full batch.
        Mini-batch is much faster (20-50x) but may have slightly lower quality.
        Recommended for datasets > 100K samples or when speed is critical.

    batch_size : int, default=500
        Number of sequences to sample per mini-batch iteration.
        Only used when use_minibatch=True. Larger values are more stable
        but slower. Smaller values are faster but may be less accurate.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_projections, window_size)
        Cluster centers in the projected space (sorted 1D projections).

    labels_ : ndarray of shape (n_samples - 1,)
        Cluster labels for each time point (length is n_samples - 1 due to differencing).

    inertia_ : float
        Sum of squared distances of sequences to their closest cluster center.

    n_iter_ : int
        Number of iterations performed in the best run.

    projection_directions_ : ndarray of shape (n_projections, n_features)
        Unit vectors used for random projections.

    n_sequences_ : int
        Number of sequences extracted from the time series.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(200, 2).cumsum(axis=0)
    >>> swk = SlicedWassersteinKMeans(n_clusters=2, window_size=20, lifting_size=5)
    >>> swk.fit(X)
    >>> labels = swk.labels_
    >>> probs = swk.predict_proba(X)
    """

    def __init__(
        self,
        n_clusters: int = 2,
        window_size: int = 35,
        lifting_size: int = 7,
        n_projections: int = 9,
        p: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_init: int = 10,
        standardize: bool = False,
        temperature: float = 1.0,
        chunk_size: int = 1000,
        use_minibatch: bool = False,
        batch_size: int = 500,
    ):
        # Store parameters
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.lifting_size = lifting_size
        self.n_projections = n_projections
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.standardize = standardize
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate hyperparameters."""
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")

        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")

        if self.lifting_size <= 0:
            raise ValueError(f"lifting_size must be > 0, got {self.lifting_size}")

        if self.lifting_size >= self.window_size:
            raise ValueError(
                f"lifting_size ({self.lifting_size}) must be < window_size ({self.window_size})"
            )

        if self.n_projections <= 0:
            raise ValueError(f"n_projections must be > 0, got {self.n_projections}")

        if self.p not in [1, 2]:
            raise ValueError(f"p must be 1 or 2, got {self.p}")

        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.max_iter}")

        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol}")

        if self.n_init <= 0:
            raise ValueError(f"n_init must be > 0, got {self.n_init}")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

    def __repr__(self) -> str:
        return (
            f"SlicedWassersteinKMeans(n_clusters={self.n_clusters}, "
            f"window_size={self.window_size}, lifting_size={self.lifting_size}, "
            f"n_projections={self.n_projections}, p={self.p}, chunk_size={self.chunk_size})"
        )

    def _generate_projection_directions(
        self, n_features: int, random_state: np.random.RandomState
    ) -> np.ndarray:
        """
        Generate uniformly distributed unit vectors on the sphere.

        Uses Gaussian random vectors normalized to unit length.

        Returns
        -------
        directions : ndarray of shape (n_projections, n_features)
            Unit vectors for random projections.
        """
        directions = random_state.randn(self.n_projections, n_features)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        return directions / norms

    def _compute_returns(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log returns from price data with optional standardization.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Price or value time series.

        Returns
        -------
        returns : ndarray of shape (n_samples - 1, n_features)
            Log returns, optionally standardized.
        """
        # Compute log returns
        if np.any(X <= 0):
            raise ValueError(
                "SWKMeans requires strictly positive input values for log-return computation. "
                f"Found {int(np.sum(X <= 0))} non-positive value(s)."
            )
        returns = np.diff(np.log(X), axis=0)

        if self.standardize:
            # Standardize coordinate-wise with safety for zero variance
            mean = np.mean(returns, axis=0)
            std = np.std(returns, axis=0)

            # Avoid division by zero
            std[std < 1e-10] = 1.0

            returns = (returns - mean) / std

        return returns

    def _extract_sequences(self, returns: np.ndarray) -> np.ndarray:
        """
        Apply sliding window lifting transformation efficiently.

        Uses NumPy's stride tricks for memory-efficient window extraction.

        Parameters
        ----------
        returns : ndarray of shape (n_samples, n_features)
            Return time series.

        Returns
        -------
        sequences : ndarray of shape (n_sequences, window_size, n_features)
            Extracted sequences.
        """
        n_samples, n_features = returns.shape

        # Calculate number of sequences
        n_sequences = (n_samples - self.window_size) // self.lifting_size + 1

        if n_sequences <= 0:
            raise ValueError(
                f"Time series too short. Got {n_samples} samples, "
                f"need at least {self.window_size}"
            )

        # Pre-allocate output
        sequences = np.empty(
            (n_sequences, self.window_size, n_features), dtype=returns.dtype
        )

        # Extract sequences efficiently
        for i in range(n_sequences):
            start = i * self.lifting_size
            end = start + self.window_size
            sequences[i] = returns[start:end]

        return sequences

    def _project_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Project all sequences onto all directions and sort in one vectorized operation.

        Parameters
        ----------
        sequences : ndarray of shape (n_sequences, window_size, n_features)
            Input sequences.

        Returns
        -------
        projections : ndarray of shape (n_sequences, n_projections, window_size)
            Sorted 1D projections.
        """
        # Shape: (n_sequences, window_size, n_projections)
        projections = np.einsum("swf,pf->swp", sequences, self.projection_directions_)

        # Sort along window dimension
        projections = np.sort(projections, axis=1)

        # Reorder to (n_sequences, n_projections, window_size)
        return np.transpose(projections, (0, 2, 1))

    def _wasserstein_distance_1d_batch(
        self, mu: np.ndarray, nu: np.ndarray
    ) -> np.ndarray:
        """
        Compute 1D Wasserstein distances between batches of sorted arrays.

        Parameters
        ----------
        mu : ndarray of shape (..., n_atoms)
            First batch of sorted 1D measures.
        nu : ndarray of shape (..., n_atoms)
            Second batch of sorted 1D measures.

        Returns
        -------
        distances : ndarray of shape (...)
            Wasserstein distances.
        """
        diff = mu - nu

        if self.p == 1:
            return np.mean(np.abs(diff), axis=-1)
        else:  # p == 2
            return np.sqrt(np.mean(diff**2, axis=-1))

    def _compute_distances_matrix(
        self, seq_projections: np.ndarray, centroid_projections: np.ndarray
    ) -> np.ndarray:
        """
        Compute distance matrix between sequences and centroids efficiently using chunking.

        This method processes sequences in chunks to avoid memory explosion and uses
        Numba JIT compilation for significant speedup.

        Parameters
        ----------
        seq_projections : ndarray of shape (n_sequences, n_projections, window_size)
            Projected sequences.
        centroid_projections : ndarray of shape (n_clusters, n_projections, window_size)
            Projected centroids.

        Returns
        -------
        distances : ndarray of shape (n_sequences, n_clusters)
            Sliced Wasserstein distances.
        """
        n_sequences = seq_projections.shape[0]
        n_clusters = centroid_projections.shape[0]
        distances = np.empty((n_sequences, n_clusters), dtype=np.float64)

        # Process in chunks to avoid memory explosion
        for start_idx in range(0, n_sequences, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_sequences)
            chunk = seq_projections[start_idx:end_idx]

            # Use Numba-accelerated distance computation
            distances[start_idx:end_idx] = _compute_distances_chunked_numba(
                chunk, centroid_projections, self.p, self.n_projections
            )

        return distances

    def _wasserstein_barycenter_1d(self, measures: np.ndarray) -> np.ndarray:
        """
        Compute 1D Wasserstein barycenter for a batch of sorted measures using Numba.

        Parameters
        ----------
        measures : ndarray of shape (n_measures, n_atoms)
            Sorted 1D measures.

        Returns
        -------
        barycenter : ndarray of shape (n_atoms,)
            Wasserstein barycenter.
        """
        return _wasserstein_barycenter_1d_numba(measures, self.p)

    def _update_centroids(
        self, seq_projections: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Update cluster centroids based on assignments.

        Parameters
        ----------
        seq_projections : ndarray of shape (n_sequences, n_projections, window_size)
            Projected sequences.
        labels : ndarray of shape (n_sequences,)
            Cluster assignments.

        Returns
        -------
        new_centroids : ndarray of shape (n_clusters, n_projections, window_size)
            Updated centroids.
        """
        new_centroids = np.empty(
            (self.n_clusters, self.n_projections, self.window_size),
            dtype=seq_projections.dtype,
        )

        for k in range(self.n_clusters):
            mask = labels == k
            cluster_projs = seq_projections[mask]

            if len(cluster_projs) == 0:
                # Empty cluster - reinitialize to farthest sequence
                warnings.warn(
                    f"Cluster {k} is empty. Reinitializing to random sequence.",
                    RuntimeWarning,
                )
                # Find sequence farthest from any centroid
                if hasattr(self, "_temp_centroids"):
                    distances = self._compute_distances_matrix(
                        seq_projections, self._temp_centroids
                    )
                    min_dists = np.min(distances, axis=1)
                    farthest_idx = np.argmax(min_dists)
                    new_centroids[k] = seq_projections[farthest_idx]
                else:
                    # Fallback to random
                    random_idx = np.random.randint(len(seq_projections))
                    new_centroids[k] = seq_projections[random_idx]
            else:
                # Compute barycenter for each projection
                for j in range(self.n_projections):
                    new_centroids[k, j] = self._wasserstein_barycenter_1d(
                        cluster_projs[:, j, :]
                    )

        return new_centroids

    def _assign_clusters(
        self, seq_projections: np.ndarray, centroid_projections: np.ndarray
    ) -> np.ndarray:
        """
        Assign sequences to nearest centroids.

        Parameters
        ----------
        seq_projections : ndarray of shape (n_sequences, n_projections, window_size)
            Projected sequences.
        centroid_projections : ndarray of shape (n_clusters, n_projections, window_size)
            Projected centroids.

        Returns
        -------
        labels : ndarray of shape (n_sequences,)
            Cluster assignments.
        """
        distances = self._compute_distances_matrix(
            seq_projections, centroid_projections
        )
        return np.argmin(distances, axis=1)

    def _compute_inertia(
        self,
        seq_projections: np.ndarray,
        labels: np.ndarray,
        centroid_projections: np.ndarray,
    ) -> float:
        """
        Compute within-cluster sum of squared distances.

        Parameters
        ----------
        seq_projections : ndarray of shape (n_sequences, n_projections, window_size)
            Projected sequences.
        labels : ndarray of shape (n_sequences,)
            Cluster assignments.
        centroid_projections : ndarray of shape (n_clusters, n_projections, window_size)
            Projected centroids.

        Returns
        -------
        inertia : float
            Total within-cluster sum of squared distances.
        """
        distances = self._compute_distances_matrix(
            seq_projections, centroid_projections
        )

        # Select distance to assigned cluster for each sequence
        assigned_distances = distances[np.arange(len(labels)), labels]

        return np.sum(assigned_distances**2)

    def _check_convergence(
        self, old_centroids: Optional[np.ndarray], new_centroids: np.ndarray
    ) -> bool:
        """
        Check convergence based on relative centroid movement.

        Parameters
        ----------
        old_centroids : ndarray of shape (n_clusters, n_projections, window_size) or None
            Previous centroids.
        new_centroids : ndarray of shape (n_clusters, n_projections, window_size)
            Updated centroids.

        Returns
        -------
        converged : bool
            True if converged, False otherwise.
        """
        if old_centroids is None:
            return False

        # Compute relative change
        diff = np.abs(new_centroids - old_centroids)
        total_movement = np.sum(diff)
        scale = np.sum(np.abs(new_centroids)) + 1e-10

        relative_change = total_movement / scale

        return relative_change < self.tol

    def _initialize_centroids(
        self, seq_projections: np.ndarray, random_state: np.random.RandomState
    ) -> np.ndarray:
        """
        Initialize centroids using k-means++ style initialization.

        Parameters
        ----------
        seq_projections : ndarray of shape (n_sequences, n_projections, window_size)
            Projected sequences.
        random_state : RandomState
            Random state for reproducibility.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_projections, window_size)
            Initial centroids.
        """
        n_sequences = len(seq_projections)
        centroids = np.empty(
            (self.n_clusters, self.n_projections, self.window_size),
            dtype=seq_projections.dtype,
        )

        # Choose first centroid randomly
        first_idx = random_state.randint(n_sequences)
        centroids[0] = seq_projections[first_idx]

        # Choose remaining centroids with probability proportional to distance squared
        for k in range(1, self.n_clusters):
            # Compute distances to existing centroids
            distances = self._compute_distances_matrix(seq_projections, centroids[:k])
            min_distances = np.min(distances, axis=1)

            # Compute probabilities
            probs = min_distances**2
            probs /= np.sum(probs)

            # Sample next centroid
            next_idx = random_state.choice(n_sequences, p=probs)
            centroids[k] = seq_projections[next_idx]

        return centroids

    def _single_fit(
        self, seq_projections: np.ndarray, random_state: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Single k-means run with smart initialization.

        Parameters
        ----------
        seq_projections : ndarray of shape (n_sequences, n_projections, window_size)
            Projected sequences.
        random_state : RandomState
            Random state for reproducibility.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_projections, window_size)
            Final centroids.
        labels : ndarray of shape (n_sequences,)
            Final cluster assignments.
        inertia : float
            Final inertia.
        n_iter : int
            Number of iterations performed.
        """
        # Initialize centroids with k-means++ style
        centroids = self._initialize_centroids(seq_projections, random_state)
        old_centroids = None

        for iteration in range(self.max_iter):
            # Store for empty cluster handling
            self._temp_centroids = centroids

            # Assign sequences to clusters
            labels = self._assign_clusters(seq_projections, centroids)

            # Update centroids
            new_centroids = self._update_centroids(seq_projections, labels)

            # Check convergence
            if self._check_convergence(old_centroids, new_centroids):
                centroids = new_centroids
                break

            old_centroids = centroids
            centroids = new_centroids

        # Compute final inertia
        labels = self._assign_clusters(seq_projections, centroids)
        inertia = self._compute_inertia(seq_projections, labels, centroids)

        # Clean up temporary attribute
        delattr(self, "_temp_centroids")

        return centroids, labels, inertia, iteration + 1

    def _single_fit_minibatch(
        self, seq_projections: np.ndarray, random_state: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Single mini-batch k-means run for faster training on large datasets.

        Instead of using all sequences per iteration, samples random mini-batches
        and updates centroids incrementally with a learning rate.

        Parameters
        ----------
        seq_projections : ndarray of shape (n_sequences, n_projections, window_size)
            Projected sequences.
        random_state : RandomState
            Random state for reproducibility.

        Returns
        -------
        centroids : ndarray of shape (n_clusters, n_projections, window_size)
            Final centroids.
        labels : ndarray of shape (n_sequences,)
            Final cluster assignments (computed on all data).
        inertia : float
            Final inertia (computed on all data).
        n_iter : int
            Number of iterations performed.
        """
        n_sequences = len(seq_projections)

        # Initialize centroids with k-means++ style
        centroids = self._initialize_centroids(seq_projections, random_state)

        # Track counts for each cluster (for weighted updates)
        cluster_counts = np.zeros(self.n_clusters, dtype=np.float64)

        for iteration in range(self.max_iter):
            # Sample mini-batch
            batch_size_actual = min(self.batch_size, n_sequences)
            batch_indices = random_state.choice(
                n_sequences, size=batch_size_actual, replace=False
            )
            batch_projections = seq_projections[batch_indices]

            # Assign batch to clusters
            batch_labels = self._assign_clusters(batch_projections, centroids)

            # Update centroids using mini-batch with learning rate
            # Learning rate decreases over iterations for stability
            learning_rate = 1.0 / (iteration + 1)

            for k in range(self.n_clusters):
                mask = batch_labels == k
                batch_cluster = batch_projections[mask]

                if len(batch_cluster) == 0:
                    continue

                # Compute barycenter for this cluster's batch samples
                new_centroid_k = np.empty(
                    (self.n_projections, self.window_size), dtype=seq_projections.dtype
                )
                for j in range(self.n_projections):
                    new_centroid_k[j] = self._wasserstein_barycenter_1d(
                        batch_cluster[:, j, :]
                    )

                # Incremental update with learning rate
                # More samples → more weight on new centroid
                eta = (
                    learning_rate
                    * len(batch_cluster)
                    / (cluster_counts[k] + len(batch_cluster))
                )
                centroids[k] = (1 - eta) * centroids[k] + eta * new_centroid_k
                cluster_counts[k] += len(batch_cluster)

        # Compute final assignments and inertia on full dataset
        labels = self._assign_clusters(seq_projections, centroids)
        inertia = self._compute_inertia(seq_projections, labels, centroids)

        return centroids, labels, inertia, iteration + 1

    def _map_sequence_labels_to_timeseries(
        self, sequence_labels: np.ndarray, n_returns: int
    ) -> np.ndarray:
        """
        Map sequence labels to time series labels using majority voting.

        Parameters
        ----------
        sequence_labels : ndarray of shape (n_sequences,)
            Labels for each sequence.
        n_returns : int
            Length of the return time series.

        Returns
        -------
        timeseries_labels : ndarray of shape (n_returns,)
            Labels for each time point.
        """
        timeseries_labels = np.zeros(n_returns, dtype=np.int32)

        # Pre-compute which sequences cover each time point
        for i in range(n_returns):
            # Find sequences that contain this time point
            seq_start = max(0, (i - self.window_size + 1) // self.lifting_size)
            seq_end = min(len(sequence_labels), i // self.lifting_size + 1)

            if seq_end > seq_start:
                # Get labels from relevant sequences
                relevant_labels = sequence_labels[seq_start:seq_end]

                # Majority voting using bincount
                timeseries_labels[i] = np.argmax(np.bincount(relevant_labels))

        return timeseries_labels

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the Sliced Wasserstein k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (time series values, not returns).

        y : Ignored
            Not used, present for API consistency with sklearn.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, ensure_2d=True, dtype=np.float64, ensure_all_finite=True)

        if X.shape[0] < self.window_size + 1:
            raise ValueError(
                f"Time series too short. Need at least {self.window_size + 1} samples "
                f"for window_size={self.window_size}, got {X.shape[0]}"
            )

        random_state = check_random_state(self.random_state)

        # Generate projection directions
        self.projection_directions_ = self._generate_projection_directions(
            X.shape[1], random_state
        )

        # Compute returns
        returns = self._compute_returns(X)

        # Extract sequences
        sequences = self._extract_sequences(returns)
        self.n_sequences_ = len(sequences)

        if self.n_sequences_ < self.n_clusters:
            raise ValueError(
                f"Not enough sequences ({self.n_sequences_}) for "
                f"{self.n_clusters} clusters. Reduce n_clusters or increase data length."
            )

        # Project all sequences once (memory efficient, computed on-demand)
        seq_projections = self._project_sequences(sequences)

        # Free original sequences to save memory
        del sequences

        # Multiple random initializations
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        # Choose fitting method based on use_minibatch flag
        fit_method = (
            self._single_fit_minibatch if self.use_minibatch else self._single_fit
        )

        for init_idx in range(self.n_init):
            # Create independent random state for each initialization
            init_seed = random_state.randint(2**31)
            init_random_state = np.random.RandomState(init_seed)

            try:
                centroids, seq_labels, inertia, n_iter = fit_method(
                    seq_projections, init_random_state
                )

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centroids = centroids
                    best_labels = seq_labels
                    best_n_iter = n_iter
            except Exception as e:
                warnings.warn(
                    f"Initialization {init_idx + 1} failed: {str(e)}", RuntimeWarning
                )
                continue

        if best_centroids is None:
            raise RuntimeError("All initializations failed")

        # Set final attributes
        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        # Map sequence labels to time series labels
        self.labels_ = self._map_sequence_labels_to_timeseries(
            best_labels, len(returns)
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples - 1,)
            Cluster labels for each time point (length is n_samples - 1 due to differencing).
        """
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True, dtype=np.float64, ensure_all_finite=True)

        if X.shape[0] < self.window_size + 1:
            raise ValueError(
                f"Time series too short. Need at least {self.window_size + 1} samples"
            )

        if X.shape[1] != self.projection_directions_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but model was trained with "
                f"{self.projection_directions_.shape[1]} features"
            )

        # Compute returns
        returns = self._compute_returns(X)

        # Extract sequences
        sequences = self._extract_sequences(returns)

        # Project sequences
        seq_projections = self._project_sequences(sequences)

        # Assign to clusters
        sequence_labels = self._assign_clusters(seq_projections, self.cluster_centers_)

        # Map to time series labels
        return self._map_sequence_labels_to_timeseries(sequence_labels, len(returns))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster membership probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        probabilities : ndarray of shape (n_samples - 1, n_clusters)
            Probability of each time point belonging to each cluster.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True, dtype=np.float64, ensure_all_finite=True)

        if X.shape[0] < self.window_size + 1:
            raise ValueError(
                f"Time series too short. Need at least {self.window_size + 1} samples"
            )

        if X.shape[1] != self.projection_directions_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but model was trained with "
                f"{self.projection_directions_.shape[1]} features"
            )

        # Compute returns
        returns = self._compute_returns(X)

        # Extract sequences
        sequences = self._extract_sequences(returns)

        # Project sequences
        seq_projections = self._project_sequences(sequences)

        # Compute distances to all centroids
        distances = self._compute_distances_matrix(
            seq_projections, self.cluster_centers_
        )

        # Convert to probabilities using softmax with temperature
        exp_neg_distances = np.exp(-distances / self.temperature)
        sequence_probs = exp_neg_distances / np.sum(
            exp_neg_distances, axis=1, keepdims=True
        )

        # Map sequence probabilities to time series probabilities
        return self._map_sequence_probabilities_to_timeseries(
            sequence_probs, len(returns)
        )

    def _map_sequence_probabilities_to_timeseries(
        self, sequence_probabilities: np.ndarray, n_returns: int
    ) -> np.ndarray:
        """
        Map sequence probabilities to time series probabilities efficiently.

        Parameters
        ----------
        sequence_probabilities : ndarray of shape (n_sequences, n_clusters)
            Probabilities for each sequence.
        n_returns : int
            Length of the return time series.

        Returns
        -------
        timeseries_probabilities : ndarray of shape (n_returns, n_clusters)
            Probabilities for each time point.
        """
        timeseries_probs = np.zeros((n_returns, self.n_clusters))
        counts = np.zeros(n_returns)

        # Accumulate probabilities from all sequences that cover each time point
        for seq_idx in range(len(sequence_probabilities)):
            start = seq_idx * self.lifting_size
            end = min(start + self.window_size, n_returns)
            timeseries_probs[start:end] += sequence_probabilities[seq_idx]
            counts[start:end] += 1

        # Average probabilities
        mask = counts > 0
        timeseries_probs[mask] /= counts[mask, np.newaxis]

        # For time points not covered (shouldn't happen with proper params), use uniform
        if not np.all(mask):
            timeseries_probs[~mask] = 1.0 / self.n_clusters
            warnings.warn(
                "Some time points are not covered by any sequence. Using uniform probabilities.",
                RuntimeWarning,
            )

        return timeseries_probs

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model and predict cluster trend.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency.

        Returns
        -------
        trend : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_
