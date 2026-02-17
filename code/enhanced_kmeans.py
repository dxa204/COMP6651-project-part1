import numpy as np
import time


class EnhancedKMeans:
    """
    K-Means with two enhancements:
      1. Density-Aware Spread Initialization (DASI) - custom centroid seeding
      2. Vectorized distance computation and adaptive convergence
    """

    def __init__(self, k, max_iter=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.init_time_ = 0
        self.iter_time_ = 0

        if random_state is not None:
            np.random.seed(random_state)

    def _local_density(self, X, k_neighbors=10):
        """
        For each point, compute its local density as the inverse of the
        average distance to its k nearest neighbours.
        Dense regions get a high score; sparse/outlier regions get a low score.
        Time complexity: O(n^2 * d)
        """
        n = X.shape[0]
        k_neighbors = min(k_neighbors, n - 1)

        # Pairwise Euclidean distances via ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x*y^T
        norms = np.sum(X ** 2, axis=1, keepdims=True)
        dist_matrix = np.sqrt(np.maximum(norms + norms.T - 2 * X @ X.T, 0))

        # Sort each row; skip column 0 (distance to self = 0)
        sorted_dists = np.sort(dist_matrix, axis=1)
        avg_knn_dist = np.mean(sorted_dists[:, 1:k_neighbors + 1], axis=1)

        return 1.0 / (avg_knn_dist + 1e-10)

    def _initialize_centroids(self, X):
        """
        DASI: pick k initial centroids by balancing three criteria for each
        candidate point:
          - Spread     (50%): how far it is from already-chosen centroids
          - Density    (30%): how representative it is of a dense region
          - Coverage   (20%): squared spread, to further reward distant points

        First centroid is always the densest point.
        Subsequent centroids maximise the combined score above.
        Time complexity: O(n^2 * d + k^2 * n * d)
        """
        n, d = X.shape
        centroids = np.zeros((self.k, d))
        chosen = []

        # Compute and normalise density scores once
        density = self._local_density(X)
        density = (density - density.min()) / (density.max() - density.min() + 1e-10)

        # First centroid: the point with the highest local density
        first = int(np.argmax(density))
        centroids[0] = X[first]
        chosen.append(first)

        for i in range(1, self.k):
            # Distance from every point to its nearest already-chosen centroid
            dist_to_chosen = np.column_stack([
                np.linalg.norm(X - centroids[j], axis=1) for j in range(i)
            ])
            min_dist = np.min(dist_to_chosen, axis=1)

            # Normalise to [0, 1]
            if min_dist.max() > 0:
                norm_dist = min_dist / min_dist.max()
            else:
                norm_dist = min_dist

            score = 0.5 * norm_dist + 0.3 * density + 0.2 * norm_dist ** 2

            # Prevent re-selecting an already chosen point
            score[chosen] = -np.inf

            next_idx = int(np.argmax(score))
            centroids[i] = X[next_idx]
            chosen.append(next_idx)

        return centroids

    def _assign(self, X):
        """
        Assign each point to its nearest centroid.
        Uses the identity ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x*c^T to avoid
        an explicit loop over centroids.
        Returns labels (n,) and distances to the assigned centroid (n,).
        Time complexity: O(n * k * d) per call.
        """
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)                # (n, 1)
        C_sq = np.sum(self.centroids ** 2, axis=1).reshape(1, -1)   # (1, k)
        dists = np.sqrt(np.maximum(X_sq + C_sq - 2 * X @ self.centroids.T, 0))  # (n, k)

        labels = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(X)), labels]
        return labels, min_dists

    def _update_centroids(self, X, labels):
        """
        Move each centroid to the mean of its assigned points.
        If a cluster is empty, reinitialise it to a random point.
        Returns the new centroids and the largest centroid shift (used for
        convergence checking).
        Time complexity: O(n * d)
        """
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            members = X[labels == i]
            if len(members) > 0:
                new_centroids[i] = members.mean(axis=0)
            else:
                new_centroids[i] = X[np.random.randint(len(X))]

        max_shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
        return new_centroids, max_shift

    def fit(self, X):
        if self.k > len(X):
            raise ValueError(f"k={self.k} cannot exceed number of samples ({len(X)})")

        t0 = time.time()
        self.centroids = self._initialize_centroids(X)
        self.init_time_ = time.time() - t0

        t0 = time.time()
        for i in range(self.max_iter):
            labels, dists = self._assign(X)
            new_centroids, max_shift = self._update_centroids(X, labels)
            self.centroids = new_centroids

            if max_shift < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.iter_time_ = time.time() - t0

        # Final assignment and SSE
        self.labels_, dists = self._assign(X)
        self.inertia_ = float(np.sum(dists ** 2))
        return self

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Call fit() before predict().")
        labels, _ = self._assign(X)
        return labels

    def get_performance_stats(self):
        return {
            'n_iterations':  self.n_iter_,
            'init_time':     self.init_time_,
            'iter_time':     self.iter_time_,
            'total_time':    self.init_time_ + self.iter_time_,
            'inertia':       self.inertia_,
        }