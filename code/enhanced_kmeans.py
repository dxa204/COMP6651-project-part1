import numpy as np
import time


class EnhancedKMeans:
    """
    K-Means with two enhancements:
      1. Initialization by Spread, Density and Amplification - our proposed solution for choosing centroid seeding
      2. Vectorized distance computation, adaptive convergence and stability measurement
    """

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 
    def __init__(self, k, max_iter=300, tol=1e-4, random_state=None):
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

        self.sse_history_          = []   # SSE after each iteration
        self.reassignment_history_ = []   # fraction of points relabelled 

        if random_state is not None:
            np.random.seed(random_state)

    def _local_density(self, X, k_neighbors=10):
        """
        For each point, compute its local density as the inverse of the average distance to its k nearest neighbours
        Dense regions get a high score; sparse/outlier regions get a low score

        Matrix multiplication X @ X.T results in (n x d) times (d x n) = n x n matrix as a result   
        Time complexity: O(n^2 * d)
        """
        n = X.shape[0]
        k_neighbors = min(k_neighbors, n - 1)

        # Pairwise Euclidean distances via ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x*y
        # Where y = x.T (Transpose) in this case
        norms = np.sum(X ** 2, axis=1, keepdims=True)
        distance_matrix = np.sqrt(np.maximum(norms + norms.T - 2 * X @ X.T, 0))

        # Sort each row and we skip column 0 beacuse distance to self = 0 (itself)
        sorted_distances = np.sort(distance_matrix, axis=1)
        avg_knn_distance = np.mean(sorted_distances[:, 1:k_neighbors + 1], axis=1)

        return 1.0 / (avg_knn_distance + 1e-10)

    def _initialize_centroids(self, X):
        """
        Initialization by Spread, Density and Amplification or (SDA for short): pick k initial centroids by balancing these 3 parameters for each
        candidate data point:
          - Spread                    (50%): how far it is from already-chosen centroids
          - Density                   (30%): how representative it is of a dense region
          - Amplification (Spread^2)  (20%): to further add weight to distant points

        So spread * 0.5 + density * 0.3 + amplification * 0.2 = next centroid

        First centroid is always the densest point
        Subsequent centroids maximise the combined score above
        Each entry (i, j) is the distance from point i to centroid j. argmin along axis=1 gives each point's nearest centroid
        Matrix multiplication is O(n^2 * d) + Compute distance for k centroid per iteration (hence k^2)
        Time complexity: O(n^2 * d + k^2 * n * d)
        """
        n, d = X.shape
        centroids = np.zeros((self.k, d))
        chosen = []

        # Compute and normalise density scores once
        density = self._local_density(X)
        density = (density - density.min()) / (density.max() - density.min() + 1e-10)

        # First centroid: the point with the highest local density using argmax 
        first = int(np.argmax(density))
        centroids[0] = X[first]
        chosen.append(first)

        for i in range(1, self.k):
            # Distance from every point to its nearest already-chosen centroid
            distance_to_chosen = np.column_stack([
                np.linalg.norm(X - centroids[j], axis=1) for j in range(i)
            ])
            min_distance = np.min(distance_to_chosen, axis=1)

            # Normalise to [0, 1]
            if min_distance.max() > 0:
                norm_distance = min_distance / min_distance.max()
            else:
                norm_distance = min_distance

            score = 0.5 * norm_distance + 0.3 * density + 0.2 * norm_distance ** 2

            # Prevent re-selecting an already chosen point
            score[chosen] = -np.inf

            next_idx = int(np.argmax(score))
            centroids[i] = X[next_idx]
            chosen.append(next_idx)

        return centroids

    def _assign(self, X):
        """
        Assign each point to its nearest centroid
        Uses the identity ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x*c^T to avoid an explicit loop over centroids

        Each entry (i, j) is the distance from point i to centroid j
        Argmin along axis=1 gives each point's nearest centroid
        Returns labels (n,) and distances to the assigned centroid (n,)
        Time complexity: O(n * k * d) for a singular iteration
        Time complexity: O(t * n * k * d) for t iteration
        """
        X_square = np.sum(X ** 2, axis=1, keepdims=True)                                    # (n, 1)
        C_square = np.sum(self.centroids ** 2, axis=1).reshape(1, -1)                       # (1, k)
        distances = np.sqrt(np.maximum(X_square + C_square - 2 * X @ self.centroids.T, 0))  # (n, k)

        labels = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(X)), labels]
        return labels, min_distances

    def _update_centroids(self, X, labels):
        """
        Move each centroid to the mean of its assigned points
        If a cluster is empty, reinitialise it to a random point
        Returns the new centroids and the largest centroid shift (used for convergence checking).
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
        prev_labels = np.full(len(X), -1)  # All points reassigned on the first iteration and then we loop over it
        for i in range(self.max_iter):
            labels, dists = self._assign(X)

            # Compute SSE for this iteration
            sse = float(np.sum(dists ** 2))
            self.sse_history_.append(sse)

            # Record fraction of points reassigned for this iteration
            n_reassigned = int(np.sum(labels != prev_labels))
            self.reassignment_history_.append(n_reassigned / len(X))
            prev_labels = labels.copy()

            new_centroids, max_shift = self._update_centroids(X, labels)
            self.centroids = new_centroids

            if max_shift < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.iter_time_ = time.time() - t0

        # Final assignment and SSE
        self.labels_, distances = self._assign(X)
        self.inertia_ = float(np.sum(distances ** 2))
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