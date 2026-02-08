import json

DATA_DIR = "data/"
MAIN_DATA = DATA_DIR + "main_data.json"
FEEDBACK_DATA = DATA_DIR + "feedback.json"

MIN_CONFIRMATIONS = 3.0  # weighted threshold

def load(file):
    with open(file, "r") as f:
        return json.load(f)

def save(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

def run_training_pipeline():
    main = load(MAIN_DATA)
    feedback = load(FEEDBACK_DATA)

    learned = []
    remaining = []

    for item in feedback["messages"]:
        if item["count"] >= MIN_CONFIRMATIONS:
            main["messages"].append({
                "text": item["text"],
                "label": item["label"]
            })
            learned.append(item)
        else:
            remaining.append(item)

    feedback["messages"] = remaining

    save(MAIN_DATA, main)
    save(FEEDBACK_DATA, feedback)

    # Report
    print("=== TRAINING PIPELINE REPORT ===")
    print(f"Learned samples added: {len(learned)}")
    print(f"Remaining feedback: {len(remaining)}")

    if learned:
        print("\nNewly learned messages:")
        for l in learned:
            print(f"- {l['text']} → label={l['label']}")
    else:
        print("\nNo new data met learning threshold.")

if __name__ == "__main__":
    run_training_pipeline()
