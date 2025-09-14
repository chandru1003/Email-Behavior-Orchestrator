"""
Streamlit UI for Email Behavior Orchestrator
- Reads AI predictions from predictions.json
- Loads trained model.pkl for classifying new emails
- Supports decision logging & overrides
"""

import streamlit as st
import pandas as pd
import json
import os
import joblib
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
PREDICTIONS_FILE = "predictions.json"
MODEL_FILE = "model/model.pkl"
DATA_FILE = "logs/decisions_log.json"   # decision log file

ACTIONS = [
    "Close Ticket",
    "Escalate to Manager",
    "Request Additional Information",
    "Assign to Agent",
    "No Action"
]

# ---------------------------
# Init log file if missing
# ---------------------------
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

# ---------------------------
# Load predictions
# ---------------------------
if os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "r") as f:
        predictions = json.load(f)
else:
    predictions = []

# ---------------------------
# Load classifier
# ---------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Email Behavior Orchestrator", layout="wide")
st.title("📧 Email Behavior Orchestrator — Dashboard")

# Sidebar
st.sidebar.header("Controls")
show_analytics = st.sidebar.checkbox("Show Analytics (Beta)")

# ---------------------------
# Inbox Table
# ---------------------------
if predictions:
    df = pd.DataFrame([{
        "Thread ID": p["thread_id"],
        "Sender": p["sender"],
        "AI-Detected Behavior": p["predicted_behavior"],
        "Suggested Action": p["suggested_action"],
        "Decision Status": "Pending"
    } for p in predictions])
else:
    df = pd.DataFrame(columns=["Thread ID", "Sender", "AI-Detected Behavior", "Suggested Action", "Decision Status"])

st.subheader("Inbox — Threads")
if not df.empty:
    selected = st.selectbox("Select Thread ID", options=df["Thread ID"].tolist())
    st.dataframe(df, use_container_width=True)
    thread = next(p for p in predictions if p["thread_id"] == selected)
else:
    st.warning("No predictions found. Please run the classification pipeline first.")
    thread = None

# ---------------------------
# Thread Details
# ---------------------------
if thread:
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"Thread: {thread['thread_id']} — {thread['sender']}")
        st.write("**Full conversation:**")
        st.info(thread["email_text"])

    with col2:
        st.write("**AI Analysis**")
        st.metric("Detected Behavior", thread["predicted_behavior"])
        st.write("**AI Suggestion**")
        st.write(thread["suggested_action"])
        ai_conf = st.slider("AI Confidence", min_value=50, max_value=100, value=85)

    st.markdown("---")

    # ---------------------------
    # Decision Section
    # ---------------------------
    st.subheader("Decision")
    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        if st.button("✅ Approve Suggested Action"):
            entry = {
                "thread_id": thread["thread_id"],
                "ai_suggestion": thread["suggested_action"],
                "final_action": thread["suggested_action"],
                "decision_status": "Approved",
                "user_id": "Analyst User",
                "timestamp": datetime.utcnow().isoformat(),
                "ai_confidence": ai_conf
            }
            with open(DATA_FILE, "r+") as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
            st.success("Decision saved!")

    with col_b:
        override = st.selectbox("✏ Override Action", options=ACTIONS, index=0)
        if st.button("Save Override"):
            entry = {
                "thread_id": thread["thread_id"],
                "ai_suggestion": thread["suggested_action"],
                "final_action": override,
                "decision_status": "Overridden",
                "user_id": "Analyst User",
                "timestamp": datetime.utcnow().isoformat(),
                "ai_confidence": ai_conf
            }
            with open(DATA_FILE, "r+") as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
            st.success("Override saved!")

    with col_c:
        st.write("Current Decision Status: Pending")

# ---------------------------
# New Email Test (using model.pkl)
# ---------------------------
if model:
    st.markdown("---")
    st.subheader("🔍 Test New Email with Trained Classifier")
    new_email = st.text_area("Paste a new email conversation here:")

    if st.button("Classify New Email"):
        pred = model.predict([new_email])[0]
        st.write(f"**Predicted Behavior:** {pred}")
        st.write(f"**Suggested Action:** {ACTIONS[0] if pred == 'Confirmation' else 'Review Manually'}")
else:
    st.info("Classifier model.pkl not found. Train and save the model first.")

# ---------------------------
# Audit Log
# ---------------------------
st.markdown("---")
st.subheader("Audit Log")
with open(DATA_FILE, "r") as f:
    log = json.load(f)
if log:
    log_df = pd.DataFrame(log)
    st.dataframe(log_df)
else:
    st.write("No decisions recorded yet.")

# ---------------------------
# Analytics
# ---------------------------
if show_analytics:
    st.markdown("---")
    st.subheader("Analytics (Demo)")
    if log:
        hist = pd.Series([e["ai_suggestion"] for e in log]).value_counts()
        st.bar_chart(hist)
    else:
        st.write("No data to show analytics.")


