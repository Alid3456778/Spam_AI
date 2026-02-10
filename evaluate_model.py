import sys
import pickle
import numpy as np
from datetime import datetime, timezone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from ml.data_loader import load_json, save_json
from ml.model import train_model


# ==============================
# PATH CONFIG
# ==============================

DATA_DIR = "data/"
MODEL_DIR = "models/"

MAIN_DATA = DATA_DIR + "main_data.json"
TEST_DATA = DATA_DIR + "test_data.json"

METRICS_FILE = DATA_DIR + "metrics.json"
METRICS_HISTORY = DATA_DIR + "metrics_history.json"
DRIFT_STATS = DATA_DIR + "drift_stats.json"
DRIFT_LIVE = DATA_DIR + "drift_live.json"

REGISTRY_FILE = MODEL_DIR + "model_registry.json"
LATEST_MODEL = MODEL_DIR + "model_latest.pkl"


# ==============================
# MODEL REGISTRY
# ==============================

def load_registry():
    try:
        return load_json(REGISTRY_FILE)
    except FileNotFoundError:
        return {
            "current_version": 0,
            "models": []
        }


def save_new_model(model, vectorizer, metrics):
    registry = load_registry()

    new_version = registry["current_version"] + 1
    model_name = f"model_v{new_version}.pkl"
    model_path = MODEL_DIR + model_name

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "vectorizer": vectorizer
        }, f)

    registry["models"].append({
        "version": new_version,
        "file": model_name,
        "metrics": metrics,
        "approved_at": datetime.now(timezone.utc).isoformat()
    })

    registry["current_version"] = new_version
    save_json(REGISTRY_FILE, registry)

    # Update latest pointer
    with open(LATEST_MODEL, "wb") as f:
        pickle.dump({
            "model": model,
            "vectorizer": vectorizer
        }, f)


# ==============================
# EVALUATION
# ==============================

def evaluate_model(model, vectorizer):
    test_data = load_json(TEST_DATA)["messages"]

    texts = [m["text"] for m in test_data]
    y_true = [m["label"] for m in test_data]

    X_test = vectorizer.transform(texts)
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4)
    }


# ==============================
# METRICS LOGGING
# ==============================

def log_metrics(metrics):
    history = load_json(METRICS_HISTORY)
    history["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics
    })
    save_json(METRICS_HISTORY, history)


# ==============================
# DRIFT BASELINE (TF-IDF ONLY)
# ==============================

def save_drift_baseline(vectorizer, texts):
    tfidf = None
    for name, transformer in vectorizer.transformer_list:
        if name == "tfidf":
            tfidf = transformer
            break

    if tfidf is None:
        raise RuntimeError("TF-IDF transformer not found in FeatureUnion")

    X_text = tfidf.transform(texts)
    mean_vec = np.asarray(X_text.mean(axis=0)).ravel()

    baseline = {
        "vector_size": int(mean_vec.shape[0]),
        "baseline_mean": mean_vec.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    save_json(DRIFT_STATS, baseline)

    # 🔥 CRITICAL: reset live drift vectors
    save_json(DRIFT_LIVE, {"vectors": []})


# ==============================
# MAIN PIPELINE
# ==============================

def main():
    train_data = load_json(MAIN_DATA)["messages"]
    texts = [m["text"] for m in train_data]

    # Train candidate model
    model, vectorizer = train_model(train_data)

    # Evaluate
    new_metrics = evaluate_model(model, vectorizer)

    old_metrics = load_json(METRICS_FILE)
    old_f1 = float(old_metrics.get("f1", 0.0))
    new_f1 = float(new_metrics["f1"])

    print("=== MODEL EVALUATION REPORT ===")
    print(f"Previous F1 : {old_f1}")
    print(f"New F1      : {new_f1}")
    print(f"Accuracy    : {new_metrics['accuracy']}")
    print(f"Precision   : {new_metrics['precision']}")
    print(f"Recall      : {new_metrics['recall']}")

    # ==============================
    # QUALITY GATE (F1-BASED)
    # ==============================

    if new_f1 >= old_f1:
        print("✅ Model PASSED quality gate")

        save_json(METRICS_FILE, new_metrics)
        log_metrics(new_metrics)
        save_new_model(model, vectorizer, new_metrics)
        save_drift_baseline(vectorizer, texts)

        print("📦 New model version approved & deployed")
        sys.exit(0)

    else:
        print("❌ Model FAILED quality gate")
        print("Rollback remains active")
        sys.exit(1)


# ==============================
# ENTRY
# ==============================

if __name__ == "__main__":
    main()
