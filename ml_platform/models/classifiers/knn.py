from collections import Counter
from typing import Any

from .base_classifiers import BaseClassifier

class KNN(BaseClassifier):
    """
    K-Nearest Neighbors (KNN) classifier implementation.
    """

    def __init__(self, n_neighbors: int = 5):
        self._is_fitted = False
        self.n_neighbors = 5
        self.set_params(n_neighbors=n_neighbors)
        self._classes_: list[Any] | None = None

    def fit(self, X, y):
        """Store the training samples and their labels."""

        if self.is_fitted:
            raise RuntimeError("KNN classifier has already been fitted")

        X_train = [list(sample) for sample in X]
        y_train = list(y)

        if not X_train:
            raise ValueError("X must contain at least one training sample")
        if len(X_train) != len(y_train):
            raise ValueError("X and y must contain the same number of samples")

        n_features = len(X_train[0])
        if n_features == 0 or any(len(sample) != n_features for sample in X_train):
            raise ValueError("All samples must have the same non-zero number of features")

        self.X_train = X_train
        self.y_train = y_train
        # Preserve first-seen order so probability columns and tie-breaking are stable.
        self._classes_ = list(dict.fromkeys(y_train))
        self._n_features = n_features
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict labels for a collection of samples."""
        probabilities = self.predict_proba(X)
        return [
            self.classes_[max(range(len(row)), key=row.__getitem__)]
            for row in probabilities
        ]

    def predict_proba(self, X):
        """Return class probabilities in ``classes_`` order for each sample."""
        self._check_is_fitted()
        return [self._predict_proba(sample) for sample in X]

    def get_params(self) -> dict[str, int]:
        """Return the classifier's hyperparameters."""
        return {"n_neighbors": self.n_neighbors}

    def set_params(self, **params) -> "KNN":
        """Update the classifier's hyperparameters."""

        unknown = set(params) - {"n_neighbors"}
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")

        n_neighbors = params.get("n_neighbors", self.n_neighbors)
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")

        if self.n_neighbors != n_neighbors:
            self._is_fitted = False
        self.n_neighbors = n_neighbors
        return self

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def classes_(self) -> list[Any]:
        self._check_is_fitted()
        classes = self._classes_
        if classes is None:
            raise RuntimeError("KNN classifier has not been fitted")
        return classes

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("KNN classifier has not been fitted")

    def _predict_proba(self, x):
        x = list(x)
        if len(x) != self._n_features:
            raise ValueError(f"Each sample must have {self._n_features} features")

        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = sorted(range(len(distances)), key=distances.__getitem__)[:self.n_neighbors]
        votes = Counter(self.y_train[i] for i in k_indices)
        return [votes[label] / len(k_indices) for label in self.classes_]

    def _predict(self, x):
        """Predict the label for one sample."""
        probabilities = self._predict_proba(x)
        return self.classes_[max(range(len(probabilities)), key=probabilities.__getitem__)]

    def _euclidean_distance(self, x1, x2):
        """Compute the Euclidean distance between two samples."""
        return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5
