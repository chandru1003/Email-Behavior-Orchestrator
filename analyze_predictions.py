
import json
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

INPUT_FILE = "predictions.json"
MODEL_FILE = "model/model.pkl"

BEHAVIOR_TO_ACTION = {
    "Confirmation": "Close Ticket",
    "Objection": "Request Clarification",
    "Escalation": "Forward to Manager",
    "New Information": "Update Records",
    "Unknown": "Review Manually"
}

def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def analyze(df: pd.DataFrame):
    print(" Behavior Distribution ")
    print(df["predicted_behavior"].value_counts())

    print("\n Suggested Actions")
    print(df["suggested_action"].value_counts())

    print("\n Sample Records ")
    print(df.head(5).to_string())

def train_classifier(df: pd.DataFrame):
    X = df["email_text"].fillna("")
    y = df["predicted_behavior"]
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    # Save model
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\n Classifier trained and saved to {MODEL_FILE}")
    return pipeline

def load_classifier():
    return joblib.load(MODEL_FILE)

def predict_new(pipeline, email_text: str):
    pred_behavior = pipeline.predict([email_text])[0]
    action = BEHAVIOR_TO_ACTION.get(pred_behavior, "Review Manually")
    print("\n=== Prediction for New Email ===")
    print("Email:", email_text)
    print("Predicted Behavior:", pred_behavior)
    print("Suggested Action:", action)

def main():
    df = load_data()
    analyze(df)

    # Train and save model
    pipeline = train_classifier(df)

    # Example test email
    test_email = (
        "From: client@company.com\n"
        "To: support@service.com\n"
        "Subject: Booking Confirmed\n\n"
        "Thank you for confirming my reservation."
    )
    predict_new(pipeline, test_email)

    # Example loading model from disk
    print("\n Reloading saved model from disk...")
    loaded_model = load_classifier()
    predict_new(loaded_model, "Please escalate this issue to your manager.")

if __name__ == "__main__":
    main()
