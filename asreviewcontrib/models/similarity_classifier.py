import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def _l2_norm(x):
    return np.sqrt(np.sum(x**2))


def _div_norm(x):
    norm_value = _l2_norm(x)
    if norm_value > 0:
        return x * (1.0 / norm_value)
    else:
        return x


class SimilarClassifier:
    def __init__(
        self,
        similarity_metric="cosine",  # dot_product, euclidean_dist
        combine_strategy="mean",
        proba_mode="relevant",  # relevant_irrelevant_mean, relevant_irrelevant_prod
    ):
        self.similarity_metric = similarity_metric
        self.combine_strategy = combine_strategy
        self.proba_mode = proba_mode
        self.relevant = None
        self.irrelevant = None
        self.relevant_resultant = None
        self.irrelevant_resultant = None

    def fit(self, X, y):
        """Fit the model to the data."""
        # Checking sparsity (TFIDF features)
        if sparse.isspmatrix_csr(X):
            X = X.toarray()

        self.relevant = X[y == 1, :]
        self.relevant_resultant = self._combine_features(self.relevant)

        if self.proba_mode in ["relevant_irrelevant_mean", "relevant_irrelevant_prod"]:
            self.irrelevant = X[y == 0, :]
            self.irrelevant_resultant = self._combine_features(self.irrelevant)

    def predict_proba(self, X):
        """Get the inclusion probability for each sample."""
        # Checking sparsity (TFIDF features)
        if sparse.isspmatrix_csr(X):
            X = X.toarray()

        if self.similarity_metric == "cosine":
            return self._cosine(X)
        if self.similarity_metric == "dot_product":
            return self._dot_product(X)
        if self.similarity_metric == "euclidean_dist":
            return self._euclidean_dist(X)

    def _combine_features(self, features):
        if self.combine_strategy == "mean":
            return (sum(features) / len(features)).reshape(1, -1)

    def _combine_proba(self, proba_rel, proba_irrel):
        if self.proba_mode == "relevant_irrelevant_mean":
            return sum([proba_rel, proba_irrel]) / 2
        elif self.proba_mode == "relevant_irrelevant_prod":
            return proba_rel * proba_irrel

    def _cosine(self, X):
        """Calculate probability using cosine similarity"""
        sim_relevant = cosine_similarity(self.relevant_resultant, X).reshape(-1, 1)
        proba_rel = 1 - ((sim_relevant + 1) / 2)

        if self.irrelevant_resultant:
            sim_irrelevant = cosine_similarity(self.irrelevant_resultant, X).reshape(
                -1, 1
            )
            proba_irrel = (sim_irrelevant + 1) / 2
            return self._combine_proba(proba_rel, proba_irrel)
        return proba_rel

    def _dot_product(self, X):
        """Calculate probability using dot product"""
        proba_rel = 1 - np.array(
            [np.dot(self.relevant_resultant, X[i, :]) for i in range(len(X))]
        )

        if self.irrelevant_resultant:
            proba_irrel = np.array(
                [np.dot(self.irrelevant_resultant, X[i, :]) for i in range(len(X))]
            )
            return self._combine_proba(proba_rel, proba_irrel)
        return proba_rel

    def _euclidean_dist(self, X):
        """Calculate probability using Euclidean distance"""
        self.relevant_resultant = _div_norm(self.relevant_resultant)
        X = np.array([_div_norm(np.array(X[i, :])) for i in range(len(X))])
        proba_rel = pairwise_distances(X, self.relevant_resultant, metric="euclidean")

        if self.irrelevant_resultant:
            self.irrelevant_resultant = _div_norm(self.irrelevant_resultant)
            proba_irrel = 1 - pairwise_distances(
                X, self.irrelevant_resultant, metric="euclidean"
            )
            return self._combine_proba(proba_rel, proba_irrel)
        return proba_rel
