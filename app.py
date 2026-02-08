import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- CONFIG ----------------

DATA_DIR = "data/"
MAIN_DATA = DATA_DIR + "main_data.json"
FEEDBACK_DATA = DATA_DIR + "feedback.json"
INTERNAL_REVIEW = DATA_DIR + "internal_review.json"
USER_TRUST = DATA_DIR + "user_trust.json"

# AUTO_ACCEPT_CONFIDENCE = 95
# REVIEW_THRESHOLD = 40
HIGH_CONFIDENCE = 90
LOW_CONFIDENCE = 50


# ---------------- UTIL ----------------

def load(file):
    with open(file, "r") as f:
        return json.load(f)

def save(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- MODEL ----------------

@st.cache_resource
def train_model():
    data = load(MAIN_DATA)["messages"]
    texts = [m["text"] for m in data]
    labels = [m["label"] for m in data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    return model, vectorizer

def predict(model, vectorizer, message):
    X = vectorizer.transform([message])
    probs = model.predict_proba(X)[0]
    prediction = int(probs.argmax())
    confidence = float(round(probs[prediction] * 100, 2))
    return prediction, confidence

# ---------------- FEEDBACK ----------------

def get_trust(user_id):
    data = load(USER_TRUST)
    return data["users"].get(user_id, 1.0)

def update_trust(user_id, correct):
    data = load(USER_TRUST)
    trust = data["users"].get(user_id, 1.0)

    trust += 0.1 if correct else -0.2
    trust = min(max(trust, 0.2), 2.0)

    data["users"][user_id] = round(trust, 2)
    save(USER_TRUST, data)

def save_feedback(message, label, weight):
    feedback = load(FEEDBACK_DATA)

    for item in feedback["messages"]:
        if item["text"].lower() == message.lower() and item["label"] == label:
            item["count"] += weight
            save(FEEDBACK_DATA, feedback)
            return

    feedback["messages"].append({
        "text": message,
        "label": label,
        "count": weight
    })
    save(FEEDBACK_DATA, feedback)

def send_to_internal_review(message, prediction, confidence):
    review = load(INTERNAL_REVIEW)
    review["messages"].append({
        "text": message,
        "model_prediction": prediction,
        "confidence": confidence,
        "votes": {"spam": 0, "not_spam": 0}
    })
    save(INTERNAL_REVIEW, review)

# ---------------- UI ----------------

st.set_page_config(page_title="Spam Detection AI", layout="centered")
st.title("📨 Spam Detection AI")

# Initialize session state
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'last_message' not in st.session_state:
    st.session_state.last_message = ""

user_id = "ui_user"

message = st.text_input("Enter a message", key="message_input")

if message:
    # Reset feedback state if message changed
    if st.session_state.last_message != message:
        st.session_state.feedback_given = False
        st.session_state.last_message = message
    
    model, vectorizer = train_model()
    prediction, confidence = predict(model, vectorizer, message)

    label = "🚨 SPAM" if prediction == 1 else "✅ NOT SPAM"
    st.subheader(label)
    st.write(f"**Confidence:** {confidence}%")

    # # High confidence
    # if confidence >= AUTO_ACCEPT_CONFIDENCE:
    #     st.success("Auto-accepted (high confidence)")
    
    # # Low confidence
    # elif confidence < REVIEW_THRESHOLD:
    #     send_to_internal_review(message, prediction, confidence)
    #     st.warning("Low confidence — sent for internal review")
    # -------- CONFIDENCE BUCKETS --------

    if confidence >= HIGH_CONFIDENCE:
        st.success("Auto-accepted (very high confidence)")
        # return

    elif confidence < LOW_CONFIDENCE:
        send_to_internal_review(message, prediction, confidence)
        st.warning("Low confidence — sent for internal review")

        # Allow optional user feedback
        allow_feedback = True

    # Medium confidence → user feedback
    else:
        if not st.session_state.feedback_given:
            st.info("Please confirm if this prediction is correct:")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ YES - Correct", use_container_width=True):
                    # User confirms prediction is correct
                    update_trust(user_id, True)
                    st.session_state.feedback_given = True
                    st.rerun()

            with col2:
                if st.button("❌ NO - Wrong", use_container_width=True):
                    # User says prediction is wrong
                    trust = get_trust(user_id)
                    correct_label = 0 if prediction == 1 else 1
                    save_feedback(message, correct_label, trust)
                    update_trust(user_id, False)
                    st.session_state.feedback_given = True
                    st.session_state.feedback_correct_label = correct_label
                    st.session_state.feedback_trust = trust
                    st.rerun()
        else:
            # Show feedback result after button click
            if hasattr(st.session_state, 'feedback_correct_label'):
                # User said NO
                correct_text = "NOT SPAM" if st.session_state.feedback_correct_label == 0 else "SPAM"
                st.success(f"✅ Feedback recorded! Correct label: {correct_text} 🧠")
                st.info(f"Your feedback (weighted by trust: {st.session_state.feedback_trust:.2f}) will help improve the model.")
            else:
                # User said YES
                st.success("✅ Thanks for confirming! Your trust score increased. 👍")
                st.balloons()