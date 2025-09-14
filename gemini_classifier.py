import json
import time
import re
from typing import List
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# CONFIG

GOOGLE_API_KEY = "AIzaSyBDfA2NfRysu8PQ7g1Vf0UELR8JDuPZjfI"  
INPUT_JSON = "Data/sample_emails.json"
OUTPUT_PREDICTIONS = "predictions.json"

MODEL = "gemini-2.0-flash"
CATEGORIES = ["Confirmation", "Objection", "Escalation", "New Information", "Unknown"]

BEHAVIOR_TO_ACTION = {
    "Confirmation": "Close Ticket",
    "Objection": "Request Clarification",
    "Escalation": "Forward to Manager",
    "New Information": "Update Records",
    "Unknown": "Review Manually"
}


# Utils

def extract_thread_text(msgs: List[str]) -> str:
    """Join all messages in a thread into one string."""
    if isinstance(msgs, list):
        return "\n---\n".join(msgs)
    return str(msgs)

# Gemini Classification

def gemini_classify(thread_text: str) -> str:
    prompt = (
        f"You are an AI assistant classifying email conversation threads into one of these categories: "
        f"{', '.join(CATEGORIES)}.\n\n"
        "Definitions:\n"
        "- Confirmation: Confirms, accepts, or acknowledges a request, booking, or agreement.\n"
        "- Objection: Rejects, disagrees, disputes, or raises an issue with the request.\n"
        "- Escalation: Demands higher-level intervention (manager, authority, urgent resolution).\n"
        "- New Information: Provides new data, details, or updates not present earlier.\n"
        "- Unknown: None of the above apply.\n\n"
        "Instruction: Analyze the entire conversation thread, not just the last message. "
        "Return EXACTLY one of these categories: Confirmation, Objection, Escalation, New Information, Unknown.\n\n"
        f"Thread:\n{thread_text}\n\nAnswer:"
    )

    response = model.generate_content(prompt)
    time.sleep(15)  # avoid quota spikes
    text = response.text.strip() if response and hasattr(response, "text") else "Unknown"


    print("\n[DEBUG] Raw Gemini output:")
    print(text)


    for cat in CATEGORIES:
        if re.search(rf"\b{cat.lower()}\b", text.lower()):
            return cat

    return "Unknown"

# Pipeline

def run_pipeline():
    genai.configure(api_key=GOOGLE_API_KEY)
    global model
    model = genai.GenerativeModel(MODEL)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        threads = json.load(f)
    df = pd.DataFrame(threads)
    df["email_text"] = df["messages"].apply(extract_thread_text)

    print(f"Loaded {len(df)} threads")

    labels = []
    for text in tqdm(df["email_text"], desc="Classifying with Gemini"):
        label = gemini_classify(text)
        labels.append(label)
    df["predicted_behavior"] = labels
    df["suggested_action"] = df["predicted_behavior"].map(BEHAVIOR_TO_ACTION)

    result = df[["thread_id", "sender", "email_text", "predicted_behavior", "suggested_action"]].to_dict(orient="records")
    with open(OUTPUT_PREDICTIONS, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f" Saved predictions to {OUTPUT_PREDICTIONS}")

if __name__ == "__main__":
    run_pipeline()
