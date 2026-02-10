import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


class TextPreprocessor:
    """Text preprocessing for spam detection"""

    @staticmethod
    def preprocess(text: str) -> str:
        # Lowercase
        text = text.lower()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Intentionally keep:
        # - URLs
        # - numbers
        # - symbols
        # (they are spam indicators)

        return text


def train_model(messages):
    """
    Train a probability-capable ensemble spam classifier.

    Args:
        messages: list of { "text": str, "label": int }

    Returns:
        model: sklearn-compatible classifier (supports predict_proba)
        vectorizer: fitted TfidfVectorizer
    """

    # Extract data
    texts = [m["text"] for m in messages]
    labels = [m["label"] for m in messages]

    # Preprocess
    texts = [TextPreprocessor.preprocess(t) for t in texts]

    # TF-IDF tuned for spam
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1500,
        max_df=0.85,
        min_df=1,
        sublinear_tf=True,
        lowercase=False,  # already handled
        stop_words=None
    )

    X = vectorizer.fit_transform(texts)

    # Probability-capable ensemble (IMPORTANT)
    model = VotingClassifier(
        estimators=[
            ("nb", MultinomialNB(alpha=0.1)),
            ("lr", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="liblinear",
                random_state=42
            ))
        ],
        voting="soft",  # ✅ REQUIRED for confidence
        n_jobs=-1
    )

    model.fit(X, labels)

    return model, vectorizer
