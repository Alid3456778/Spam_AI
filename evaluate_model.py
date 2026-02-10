import sys
import pickle
import numpy as np
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score

from ml.data_loader import load_json, save_json
from ml.model import train_model

DATA_DIR = "data/"
MODEL_DIR = "models/"

MAIN_DATA = DATA_DIR + "main_data.json"
TEST_DATA = DATA_DIR + "test_data.json"
METRICS_FILE = DATA_DIR + "metrics.json"
REGISTRY_FILE = MODEL_DIR + "model_registry.json"
LATEST_MODEL = MODEL_DIR + "model_latest.pkl"
METRIC_HISTORY = DATA_DIR + "metrics_history.json"


def save_drift_baseline(vectorizer, X_train):
    baseline = {
        "baseline_vocab": vectorizer.get_feature_names_out().tolist(),
        "baseline_mean": np.asarray(X_train.mean(axis=0)).ravel().tolist()
    }
    save_json("data/drift_stats.json", baseline)


# ---------------- HELPER FUNCTION ----------------

def log_metrics(accuracy):
    history = load_json(METRIC_HISTORY)
    history["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy": accuracy
    })
    save_json("data/metrics_history.json", history)


# ---------------- REGISTRY ----------------

def load_registry():
    try:
        return load_json(REGISTRY_FILE)
    except FileNotFoundError:
        return {
            "current_version": 0,
            "models": []
        }


def save_new_model(model, vectorizer, accuracy):
    registry = load_registry()

    new_version = registry["current_version"] + 1
    model_name = f"model_v{new_version}.pkl"
    model_path = MODEL_DIR + model_name

    # Save versioned model
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "vectorizer": vectorizer
        }, f)

    registry["models"].append({
        "version": new_version,
        "file": model_name,
        "accuracy": accuracy,
        "approved_at": datetime.utcnow().isoformat()
    })

    registry["current_version"] = new_version
    save_json(REGISTRY_FILE, registry)

    # Update latest pointer
    with open(LATEST_MODEL, "wb") as f:
        pickle.dump({
            "model": model,
            "vectorizer": vectorizer
        }, f)


# ---------------- EVALUATION ----------------

def evaluate_model(model, vectorizer):
    test_data = load_json(TEST_DATA)["messages"]

    X_test = vectorizer.transform([m["text"] for m in test_data])
    y_true = [m["label"] for m in test_data]
    y_pred = model.predict(X_test)

    return round(float(accuracy_score(y_true, y_pred)), 4)


# ---------------- MAIN GATE ----------------

def main():
    # Load training data
    train_data = load_json(MAIN_DATA)["messages"]

    # Train candidate model
    model, vectorizer = train_model(train_data)

    # Evaluate
    new_accuracy = evaluate_model(model, vectorizer)

    metrics = load_json(METRICS_FILE)
    # old_accuracy = metrics.get("accuracy", 0.0)
    old_accuracy = float(metrics.get("accuracy", 0.0))
    new_accuracy = float(new_accuracy)
    
    print("=== MODEL EVALUATION REPORT ===")
    print(f"Previous accuracy : {old_accuracy}")
    print(f"New accuracy      : {new_accuracy}")

    if new_accuracy >= old_accuracy:
        print("✅ Model PASSED quality gate")

        metrics["accuracy"] = new_accuracy
        save_json(METRICS_FILE, metrics)

        save_new_model(model, vectorizer, new_accuracy)
        X_train = vectorizer.transform([m["text"] for m in train_data])
        save_drift_baseline(vectorizer, X_train)

        log_metrics(new_accuracy)


        print("📦 New model version approved & deployed")
        sys.exit(0)   # PASS CI
    else:
        print("❌ Model FAILED quality gate")
        print("Rollback remains active")
        sys.exit(1)   # FAIL CI


if __name__ == "__main__":
    main()
