# ASReview Similarity Classifier Extension

This extension adds a new set of classifiers based on the similarity between the features of the
relevant records and the unlabelled records.


## Getting started

To install this extension, clone the repository to your system and then run the following command from inside the repository.

```bash
pip install -e .
```

or you can directly install it from GitHub using

```bash
pip install git+https://github.com/rohitgarud/Asreview-SimilarityClassifier.git
```

## Usage
After installation, the similarity classifier can be used as any other classifier in the simulation mode using:
```bash
asreview simulate benchmark:van_de_Schoot_2017 -m similarity -e doc2vec
```
Although the classifier can be used with TFIDF features, due to similarity measurements and the high dimensionality of TFIDF features, it is prohibitively slow for larger datasets. Hence, Doc2Vec features are recommended.

There are three different similarity metrics available, cosine similarity, dot product and Euclidean distance between the feature vectors of the resultant of relevant records and feature vectors of the unlabelled records.  

Note: This classifier was initially developed for testing the usability of different feature vectors and the similarity metrics for retrieving relevant records and potentially developing stopping criteria for screening. However, it has outperformed the default Naive Bayes and TFIDF settings in the case of many benchmark datasets. You can see the [Simulation study](https://github.com/asreview/asreview/discussions/1371), which was performed using the Similarity classifier with Doc2Vec features with the help of ASReview-Makita extension.

## License

Apache 2.0 license
