import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


class MessageBehaviorFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            t = text.lower()
            features.append([
                int(bool(re.search(r"\bfree\b|\bwin\b|\boffer\b|\bclaim\b", t))),
                int(bool(re.search(r"\bnow\b|\burgent\b|\blimited\b|\bhurry\b", t))),
                int("?" in t),
                len(t.split()),
            ])
        return np.array(features)


def train_model(messages):
    texts = [m["text"] for m in messages]
    labels = [m["label"] for m in messages]

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        sublinear_tf=True,
        min_df=2
    )

    features = FeatureUnion([
        ("tfidf", tfidf),
        ("behavior", MessageBehaviorFeatures())
    ])

    X = features.fit_transform(texts)

    # Strong linear classifier + probability calibration
    svm = LinearSVC(class_weight="balanced")
    model = CalibratedClassifierCV(svm)

    model.fit(X, labels)

    return model, features
