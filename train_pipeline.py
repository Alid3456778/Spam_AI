from ml.data_loader import load_json, save_json

DATA_DIR = "data/"
MAIN_DATA = DATA_DIR + "main_data.json"
FEEDBACK_DATA = DATA_DIR + "feedback.json"

# Minimum weighted feedback required to learn
MIN_CONFIRMATIONS = 3.0


def run_training_pipeline():
    main_data = load_json(MAIN_DATA)
    feedback_data = load_json(FEEDBACK_DATA)

    learned = []
    remaining = []

    for item in feedback_data["messages"]:
        if item["count"] >= MIN_CONFIRMATIONS:
            main_data["messages"].append({
                "text": item["text"],
                "label": item["label"]
            })
            learned.append(item)
        else:
            remaining.append(item)

    feedback_data["messages"] = remaining

    save_json(MAIN_DATA, main_data)
    save_json(FEEDBACK_DATA, feedback_data)

    # ---- REPORT ----
    print("=== TRAIN PIPELINE REPORT ===")
    print(f"New samples learned : {len(learned)}")
    print(f"Pending feedback    : {len(remaining)}")

    if learned:
        print("\nLearned messages:")
        for m in learned:
            label = "SPAM" if m["label"] == 1 else "NOT SPAM"
            print(f"- {m['text']} → {label}")
    else:
        print("\nNo feedback met learning threshold.")


if __name__ == "__main__":
    run_training_pipeline()
