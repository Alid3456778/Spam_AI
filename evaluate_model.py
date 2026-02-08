import json
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

DATA_DIR = "data/"
MODEL_DIR = "models/"

MAIN_DATA = DATA_DIR + "main_data.json"
TEST_DATA = DATA_DIR + "test_data.json"
METRICS_FILE = DATA_DIR + "metrics.json"
MODEL_FILE = MODEL_DIR + "model_latest.pkl"

# ---------------- UTIL ----------------

def load(file):
    with open(file, "r") as f:
        return json.load(f)

def save(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- TRAIN ----------------

def train_model():
    data = load(MAIN_DATA)["messages"]
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    return model, vectorizer

# ---------------- EVALUATE ----------------

def evaluate(model, vectorizer):
    test_data = load(TEST_DATA)["messages"]

    X_test = vectorizer.transform([d["text"] for d in test_data])
    y_true = [d["label"] for d in test_data]
    y_pred = model.predict(X_test)

    return round(float(accuracy_score(y_true, y_pred)), 4)

# ---------------- GATE ----------------

def main():
    model, vectorizer = train_model()
    new_accuracy = evaluate(model, vectorizer)

    metrics = load(METRICS_FILE)
    old_accuracy = metrics.get("accuracy", 0.0)

    print("=== MODEL EVALUATION REPORT ===")
    print(f"Previous accuracy : {old_accuracy}")
    print(f"New accuracy      : {new_accuracy}")

    if new_accuracy >= old_accuracy:
        print("✅ Model PASSED quality gate")

        # Save metrics
        metrics["accuracy"] = new_accuracy
        save(METRICS_FILE, metrics)

        # Save approved model
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(
                {"model": model, "vectorizer": vectorizer},
                f
            )

        sys.exit(0)   # PASS
    else:
        print("❌ Model FAILED quality gate")
        print("Deployment blocked")
        sys.exit(1)   # FAIL

if __name__ == "__main__":
    main()
