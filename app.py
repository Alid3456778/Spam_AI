import streamlit as st
import pickle
import numpy as np
import subprocess
from datetime import datetime, timezone
from ml.data_loader import load_json, save_json
from ml.predict import predict_message

# ---------------- CONFIG ----------------

RECENT_LIMIT = 200
DATA_DIR = "data/"
MODEL_DIR = "models/"

FEEDBACK_DATA = DATA_DIR + "feedback.json"
INTERNAL_REVIEW = DATA_DIR + "internal_review.json"
USER_TRUST = DATA_DIR + "user_trust.json"
LATEST_MODEL = MODEL_DIR + "model_latest.pkl"

HIGH_CONFIDENCE = 90
LOW_CONFIDENCE = 50


# ---------------- HELPER ----------------

def log_confidence(confidence):
    history = load_json("data/metrics_history.json")
    history["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": confidence
    })
    save_json("data/metrics_history.json", history)
    


def log_live_message(message, vectorizer):
    stats = load_json("data/drift_live.json")

    X = vectorizer.transform([message])
    vec = np.asarray(X.todense()).ravel().tolist()

    stats.setdefault("vectors", []).append(vec)

    # Keep last N messages only
    stats["vectors"] = stats["vectors"][-RECENT_LIMIT:]

    save_json("data/drift_live.json", stats)

# ---------------- DATA DRIFT ----------------

def check_drift(threshold=0.25):
    import numpy as np

    baseline = load_json("data/drift_stats.json")
    live = load_json("data/drift_live.json")

    if not live["vectors"]:
        return None

    baseline_mean = np.array(baseline["baseline_mean"])
    live_mean = np.mean(np.array(live["vectors"]), axis=0)

    # Cosine distance
    similarity = np.dot(baseline_mean, live_mean) / (
        np.linalg.norm(baseline_mean) * np.linalg.norm(live_mean)
    )

    drift_score = 1 - similarity
    return round(float(drift_score), 3)


# ---------------- TRUST & FEEDBACK ----------------

def get_trust(user_id):
    data = load_json(USER_TRUST)
    return data["users"].get(user_id, 1.0)

def update_trust(user_id):
    data = load_json(USER_TRUST)
    trust = data["users"].get(user_id, 1.0)
    trust = min(trust + 0.1, 2.0)
    data["users"][user_id] = round(trust, 2)
    save_json(USER_TRUST, data)

def save_feedback(message, label, weight):
    feedback = load_json(FEEDBACK_DATA)

    for item in feedback["messages"]:
        if item["text"].lower() == message.lower() and item["label"] == label:
            item["count"] += weight
            save_json(FEEDBACK_DATA, feedback)
            return

    feedback["messages"].append({
        "text": message,
        "label": label,
        "count": weight
    })
    save_json(FEEDBACK_DATA, feedback)

def send_to_internal_review(message, prediction, confidence):
    review = load_json(INTERNAL_REVIEW)
    review["messages"].append({
        "text": message,
        "model_prediction": int(prediction),
        "confidence": float(confidence),
        "votes": {"spam": 0, "not_spam": 0}
    })
    save_json(INTERNAL_REVIEW, review)

# ---------------- LOAD APPROVED MODEL ----------------

@st.cache_resource
# def load_approved_model():
#     with open(LATEST_MODEL, "rb") as f:
#         obj = pickle.load(f)
#     return obj["model"], obj["vectorizer"]
def get_model():
    if "model" not in st.session_state:
        with open(LATEST_MODEL, "rb") as f:
            obj = pickle.load(f)
        st.session_state.model = obj["model"]
        st.session_state.vectorizer = obj["vectorizer"]

    return st.session_state.model, st.session_state.vectorizer


# ---------------- UI ----------------

st.set_page_config(page_title="Spam Detection AI", layout="centered")
st.title("📨 Spam Detection AI")

# -------- ADMIN ROUTE --------
is_admin = st.query_params.get("admin") == "true"

if is_admin:
    st.divider()
    st.subheader("📊 Model Monitoring Dashboard")

    history = load_json("data/metrics_history.json")["history"]

    if history:
        accuracy_data = [h["accuracy"] for h in history if "accuracy" in h]
        confidence_data = [h["confidence"] for h in history if "confidence" in h]

        if accuracy_data:
            st.line_chart({
                "Accuracy": accuracy_data
            })

        if confidence_data:
            st.line_chart({
                "Confidence": confidence_data
            })
    else:
        st.info("No metrics recorded yet.")

# drift_score = check_drift()

# if drift_score is not None:
#     st.metric("📉 Data Drift Score", drift_score)

#     if drift_score > 0.3:
#         st.error("🚨 High data drift detected — retraining recommended")
#     elif drift_score > 0.15:
#         st.warning("⚠️ Moderate data drift detected")
#     else:
#         st.success("✅ Data distribution stable")
if is_admin:
    st.subheader("📉 Data Drift Monitoring")
    drift_score = check_drift()

    if drift_score is not None:
        st.metric("Drift Score", drift_score)

        if drift_score > 0.3:
            st.error("🚨 High drift detected — retraining recommended")
        elif drift_score > 0.15:
            st.warning("⚠️ Moderate drift detected")
        else:
            st.success("✅ Data distribution stable")



if is_admin:
    st.divider()
    st.subheader("🔐 Admin Training Panel")
    st.warning("⚠️ This runs on the SERVER using real user data")

    col1, col2 = st.columns(2)

    # 1️⃣ TRAIN (merge feedback → main_data)
    with col1:
        if st.button("🔁 Run Training Pipeline", use_container_width=True):
            with st.spinner("Running training pipeline on server..."):
                result = subprocess.run(
                    ["python", "train_pipeline.py"],
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                st.success("Training pipeline completed")
                st.code(result.stdout)
            else:
                st.error("Training pipeline failed")
                st.code(result.stderr)

    # 2️⃣ EVALUATE + VERSION
    with col2:
        if st.button("🧪 Evaluate & Approve Model", use_container_width=True):
            with st.spinner("Evaluating model quality gate..."):
                result = subprocess.run(
                    ["python", "evaluate_model.py"],
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                st.success("✅ Model approved & deployed")
                st.code(result.stdout)
            else:
                st.error("❌ Model rejected (rollback active)")
                st.code(result.stdout)


# Session state
st.session_state.setdefault("feedback_given", False)
st.session_state.setdefault("last_message", "")

user_id = "ui_user"
message = st.text_input("Enter a message")

if message:
    if st.session_state.last_message != message:
        st.session_state.feedback_given = False
        st.session_state.last_message = message

    # model, vectorizer = load_approved_model()
    model, vectorizer = get_model()

    prediction, confidence = predict_message(model, vectorizer, message)
    log_confidence(confidence)
    # log_live_message(message, vectorizer)



    label = "🚨 SPAM" if prediction == 1 else "✅ NOT SPAM"
    st.subheader(label)
    st.write(f"**Confidence:** {confidence}%")

    # -------- CONFIDENCE BUCKETS --------

    allow_feedback = False

    if confidence >= HIGH_CONFIDENCE:
        st.success("Auto-accepted (high confidence)")

    elif confidence < LOW_CONFIDENCE:
        send_to_internal_review(message, prediction, confidence)
        st.warning("Low confidence — sent for internal review")
        allow_feedback = True

    else:
        allow_feedback = True

    # -------- FEEDBACK --------

    if allow_feedback and not st.session_state.feedback_given:
        # st.info("Is this prediction correct?")
        st.info("Select the correct label:")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚨 This is SPAM"):
                trust = get_trust(user_id)
                save_feedback(message, 1, trust)
                update_trust(user_id)
                st.session_state.feedback_given = True
                st.success("Saved as SPAM 🧠")
                st.rerun()

        with col2:
            if st.button("✅ This is NOT SPAM"):
                trust = get_trust(user_id)
                save_feedback(message, 0, trust)
                update_trust(user_id)
                st.session_state.feedback_given = True
                st.success("Saved as NOT SPAM 🧠")
                st.rerun()

