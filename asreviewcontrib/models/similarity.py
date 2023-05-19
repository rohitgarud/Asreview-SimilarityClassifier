from asreview.models.classifiers.base import BaseTrainClassifier

from asreviewcontrib.models.similarity_classifier import SimilarClassifier


class SimilarityClassifier(BaseTrainClassifier):
    """Similarity classifier"""

    name = "similarity"
    label = "Similarity"

    def __init__(
        self,
        similarity_metric="cosine",  # dot_product, euclidean_dist
        combine_strategy="mean",
        proba_mode="relevant",  # relevant_irrelevant_mean, relevant_irrelevant_prod
    ):
        super(SimilarityClassifier, self).__init__()
        self._model = SimilarClassifier(
            similarity_metric=similarity_metric,
            combine_strategy=combine_strategy,
            proba_mode=proba_mode,
        )

    def fit(self, X, y):
        """Fit the model to the data."""
        self._model.fit(X, y)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
