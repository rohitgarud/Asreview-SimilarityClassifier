import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


class SimilarClassifier:
    def __init__(
        self,
        similarity_metric="cosine",  # dot_product
        combine_strategy="mean",
    ):
        self.similarity_metric = similarity_metric
        self.combine_strategy = combine_strategy
        self.relevant = None
        self.combined = None

    def fit(self, X, y):
        """Fit the model to the data."""
        # Checking sparsity (TFIDF features)
        if sparse.isspmatrix_csr(X):
            X = X.toarray()

        if self.combine_strategy == "mean":
            # Taking mean of features of relevant records
            self.relevant = X[y == 1, :]
            self.resultant = (sum(self.relevant) / len(self.relevant)).reshape(1, -1)

    def predict_proba(self, X):
        """Get the inclusion probability for each sample."""
        # Checking sparsity (TFIDF features)
        if sparse.isspmatrix_csr(X):
            X = X.toarray()

        if self.similarity_metric == "cosine":
            sim = cosine_similarity(self.resultant, X).reshape(-1, 1)
            # Mapping to 0 to 1 range and calculating as probability of 0 class
            sim = 1 - ((sim + 1) / 2)
        if self.similarity_metric == "dot_product":
            sim = np.array([np.dot(self.resultant, X[i, :]) for i in range(len(X))])
            sim = 1 - sim

        return sim
