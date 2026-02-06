import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

DATA_DIR = "data/"
MAIN_DATA = DATA_DIR + "main_data.json"
TEST_DATA = DATA_DIR + "test_data.json"
METRICS_FILE = DATA_DIR + "metrics.json"

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

def evaluate():
    model, vectorizer = train_model()

    test_data = load(TEST_DATA)["messages"]
    test_texts = [d["text"] for d in test_data]
    true_labels = [d["label"] for d in test_data]

    X_test = vectorizer.transform(test_texts)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(true_labels, predictions)
    accuracy = round(float(accuracy), 4)

    return accuracy

# ---------------- DECISION GATE ----------------

def main():
    new_accuracy = evaluate()

    metrics = load(METRICS_FILE)
    old_accuracy = metrics.get("accuracy", 0.0)

    print(f"Previous accuracy: {old_accuracy}")
    print(f"New accuracy: {new_accuracy}")

    if new_accuracy >= old_accuracy:
        print("✅ Model performance improved or maintained.")
        metrics["accuracy"] = new_accuracy
        save(METRICS_FILE, metrics)
        sys.exit(0)   # PASS
    else:
        print("❌ Model performance degraded.")
        print("Pipeline blocked.")
        sys.exit(1)   # FAIL

# ---------------- ENTRY ----------------

if __name__ == "__main__":
    main()
