"""
alternate_kmeans.py
-------------------
Implements the Section 4 variant of K-Means.

Instead of reassigning all points every iteration, only the single point
furthest from its cluster centroid is allowed to move. This reduces
churn per iteration and focuses updates on the worst-fitting point.
"""

import numpy as np


class AlternateKMeans:


    def __init__(self, k, max_iter=300, tol=1e-4, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

        if random_state is not None:
            np.random.seed(random_state)

        # Set after fit()
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.sse_history_ = []
        self.reassignment_history_ = []

    # Initialization: pick k random points as centroids, then assign all points to nearest centroid.
    def _init_centroids(self, X):
        idx = np.random.choice(len(X), self.k, replace=False)
        return X[idx].copy()

    # Assignment : Assign points to nearest centroid (initial full assignment or final predict)
    def _assign_all(self, X):


        dists = np.linalg.norm(X[:, None] - self.centroids[None], axis=2)  # (n, k)
        return np.argmin(dists, axis=1)


    # Update step: recompute all centroids first, then evaluate the furthest points
    def _update_step(self, X, labels):

        new_centroids = np.zeros_like(self.centroids)
        reassigned = 0

        # Compute all new centroids
        for i in range(self.k):
            members = np.where(labels == i)[0]

            # Revive empty clusters with a random point
            if len(members) == 0:
                new_centroids[i] = X[np.random.randint(len(X))]
            else:
                new_centroids[i] = X[members].mean(axis=0)

        # Find furthest points and reassign if a better new centroid exists
        new_labels = labels.copy()

        for i in range(self.k):
            members = np.where(labels == i)[0]
            if len(members) == 0:
                continue

            # Furthest point from the new centroid of this cluster
            dists = np.linalg.norm(X[members] - new_centroids[i], axis=1)
            furthest_idx = members[np.argmax(dists)]
            furthest_point = X[furthest_idx]

            # Check distance against all the newly computed centroids
            all_new_dists = np.linalg.norm(new_centroids - furthest_point, axis=1)
            best_cluster = int(np.argmin(all_new_dists))

            # Move it if a better cluster exists
            if best_cluster != i:
                new_labels[furthest_idx] = best_cluster
                reassigned += 1

        return new_centroids, new_labels, reassigned

    # Compute the sum of squared errors (SSE) for the current clustering
    def _compute_sse(self, X, labels):
        """Sum of squared distances from each point to its centroid."""
        return float(sum(
            np.sum((X[labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        ))

    # Fit the model to the data X, running the alternate K-Means algorithm until convergence or max iterations.
    def fit(self, X):

        self.centroids = self._init_centroids(X)
        labels = self._assign_all(X)

        for iteration in range(self.max_iter):
            new_centroids, labels, reassigned = self._update_step(X, labels)

            shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids

            self.sse_history_.append(self._compute_sse(X, labels))
            self.reassignment_history_.append(reassigned)

            if reassigned == 0 or shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.labels_ = labels
        self.inertia_ = self._compute_sse(X, labels)
        return self

    # Predict cluster labels for new data points X using the fitted centroids.
    def predict(self, X):

        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._assign_all(X)
