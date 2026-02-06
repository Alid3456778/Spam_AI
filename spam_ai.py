import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- CONFIG ----------------

DATA_DIR = "data/"
MAIN_DATA = DATA_DIR + "main_data.json"
FEEDBACK_DATA = DATA_DIR + "feedback.json"
INTERNAL_REVIEW = DATA_DIR + "internal_review.json"
USER_TRUST = DATA_DIR + "user_trust.json"

AUTO_ACCEPT_CONFIDENCE = 95
REVIEW_THRESHOLD = 70
MIN_CONFIRMATIONS = 3

# ---------------- UTILITIES ----------------

def load(file):
    with open(file, "r") as f:
        return json.load(f)

def save(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- MODEL ----------------

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
    # prediction = probs.argmax()
    # confidence = round(probs[prediction] * 100, 2)
    prediction = int(probs.argmax())
    confidence = float(round(probs[prediction] * 100, 2))
    return prediction, confidence

# ---------------- TRUST SYSTEM ----------------

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

# ---------------- FEEDBACK ----------------

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

# ---------------- INTERNAL REVIEW ----------------

def send_to_internal_review(message, prediction, confidence):
    prediction = int(prediction)
    confidence = float(confidence)
    review = load(INTERNAL_REVIEW)
    review["messages"].append({
        "text": message,
        "model_prediction": prediction,
        "confidence": confidence,
        "votes": {"spam": 0, "not_spam": 0}
    })
    save(INTERNAL_REVIEW, review)

# ---------------- NIGHTLY LEARNING ----------------

def nightly_retrain():
    main = load(MAIN_DATA)
    feedback = load(FEEDBACK_DATA)

    remaining = []

    for item in feedback["messages"]:
        if item["count"] >= MIN_CONFIRMATIONS:
            main["messages"].append({
                "text": item["text"],
                "label": item["label"]
            })
        else:
            remaining.append(item)

    feedback["messages"] = remaining

    save(MAIN_DATA, main)
    save(FEEDBACK_DATA, feedback)

# ---------------- MAIN RUN ----------------

# def run():
#     model, vectorizer = train_model()

#     user_id = "user_demo"  # placeholder for auth/session
#     message = input("Enter message: ")

#     prediction, confidence = predict(model, vectorizer, message)

#     label_text = "SPAM" if prediction == 1 else "NOT SPAM"
#     print(f"{label_text} | Confidence: {confidence}%")

#     # High confidence → auto accept
#     if confidence >= AUTO_ACCEPT_CONFIDENCE:
#         print("Auto-accepted (high confidence).")
#         return

#     # Low confidence → internal review
#     if confidence < REVIEW_THRESHOLD:
#         print("Low confidence → sent to internal review.")
#         send_to_internal_review(message, prediction, confidence)
#         return

#     # Medium confidence → ask user
#     answer = input("Is this correct? (Y/N): ").strip().upper()

#     trust = get_trust(user_id)

#     if answer == "N":
#         correct_label = 0 if prediction == 1 else 1
#         save_feedback(message, correct_label, trust)
#         update_trust(user_id, True)
#         print("Feedback recorded.")
#     else:
#         update_trust(user_id, False)
#         print("Confirmed.")

#     nightly_retrain()
def run():
    model, vectorizer = train_model()

    user_id = "user_demo"  # placeholder for auth/session
    message = input("Enter message: ")

    prediction, confidence = predict(model, vectorizer, message)

    label_text = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"{label_text} | Confidence: {confidence}%")

    # 1️⃣ High confidence → auto accept
    if confidence >= AUTO_ACCEPT_CONFIDENCE:
        print("Auto-accepted (high confidence).")
        # return

    # 2️⃣ Low confidence → internal review
    if confidence < REVIEW_THRESHOLD:
        print("Low confidence → sent to internal review.")
        send_to_internal_review(message, prediction, confidence)
        # return

    # 3️⃣ Medium confidence → ask user
    answer = input("Is this correct? (Y/N): ").strip().upper()

    trust = get_trust(user_id)

    if answer == "N":
        correct_label = 0 if prediction == 1 else 1
        save_feedback(message, correct_label, trust)
        update_trust(user_id, True)
        print("Feedback recorded.")
    else:
        update_trust(user_id, False)
        print("Thanks for confirming.")

    nightly_retrain()


# ---------------- ENTRY ----------------

if __name__ == "__main__":
    run()
