import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------
# Custom feature extractor
# ---------------------------
class MessageBehaviorFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts simple sentence-level intent signals
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for text in X:
            text_lower = text.lower()

            features.append([
                int(bool(re.search(r"\bfree\b|\bwin\b|\boffer\b|\bclaim\b", text_lower))),  # offer
                int(bool(re.search(r"\bnow\b|\burgent\b|\blimited\b|\bhurry\b", text_lower))),  # urgency
                int("?" in text_lower),  # question
                len(text.split()),  # message length
            ])

        return np.array(features)


# ---------------------------
# Training function
# ---------------------------
def train_model(messages):
    texts = [m["text"] for m in messages]
    labels = [m["label"] for m in messages]

    # Text features (context via n-grams)
    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2000,
        max_df=0.85,
        sublinear_tf=True
    )

    # Combine text meaning + behavior signals
    combined_features = FeatureUnion([
        ("tfidf", text_vectorizer),
        ("behavior", MessageBehaviorFeatures())
    ])

    X = combined_features.fit_transform(texts)

    # Probability-capable ensemble
    model = VotingClassifier(
        estimators=[
            ("nb", MultinomialNB(alpha=0.1)),
            ("lr", LogisticRegression(max_iter=1000))
        ],
        voting="soft"
    )

    model.fit(X, labels)

    return model, combined_features
