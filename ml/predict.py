import numpy as np

def predict_message(model, vectorizer, message):
    X = vectorizer.transform([message])

    # Case 1: Model supports probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prediction = int(probs.argmax())
        confidence = float(round(probs[prediction] * 100, 2))
        return prediction, confidence

    # Case 2: Model does NOT support probability (e.g. VotingClassifier hard)
    prediction = int(model.predict(X)[0])

    # Confidence fallback (safe heuristic)
    # We intentionally keep this conservative
    confidence = 60.0

    return prediction, confidence
