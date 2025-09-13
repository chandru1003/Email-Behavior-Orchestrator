import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# File to store audit log decisions
DATA_FILE = "decisions_log.json"

# Sample threads to simulate inbox data
SAMPLE_THREADS = [
    {
        "thread_id": "T1001",
        "sender": "Hotel Sunrise",
        "last_reply_time": "2025-09-12 14:22",
        "latest_reply": "We cannot confirm room availability for those dates.",
        "ai_behavior": "Objection",
        "ai_suggestion": "Request Additional Information"
    },
    {
        "thread_id": "T1002",
        "sender": "Blue Lagoon Inn",
        "last_reply_time": "2025-09-11 09:10",
        "latest_reply": "Confirmed. Please send the invoice.",
        "ai_behavior": "Confirmation",
        "ai_suggestion": "Close Ticket"
    },
    {
        "thread_id": "T1003",
        "sender": "Mountain View",
        "last_reply_time": "2025-09-10 18:05",
        "latest_reply": "This must be escalated to the manager.",
        "ai_behavior": "Escalation",
        "ai_suggestion": "Escalate to Manager"
    }
]

# List of possible override actions
ACTIONS = ["Close Ticket", "Escalate to Manager", "Request Additional Information", "Assign to Agent", "No Action"]

# Initialize log file if not present
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

# Streamlit app configuration
st.set_page_config(page_title="Email Behavior Orchestrator", layout="wide")
st.title("Email Behavior Orchestrator — Dashboard")

# Sidebar options
st.sidebar.header("Controls")
show_analytics = st.sidebar.checkbox("Show Analytics (Beta)")

# Build inbox dashboard DataFrame
df = pd.DataFrame([{
    "Thread ID": t["thread_id"],
    "Sender": t["sender"],
    "Last Reply Time": t["last_reply_time"],
    "AI-Detected Behavior": t["ai_behavior"],
    "Suggested Action": t["ai_suggestion"],
    "Decision Status": "Pending"
} for t in SAMPLE_THREADS])

# Show inbox
st.subheader("Inbox — Threads")
selected = st.selectbox("Select Thread ID", options=df["Thread ID"].tolist())

st.dataframe(df, use_container_width=True)

# Show details of selected thread
thread = next(t for t in SAMPLE_THREADS if t["thread_id"] == selected)

st.markdown("---")
col1, col2 = st.columns([3,1])
with col1:
    st.subheader(f"Thread: {thread['thread_id']} — {thread['sender']}")
    st.write("**Full conversation (simulated):**")
    st.info("Outbound: Hi, are these dates available?\nHotel: " + thread['latest_reply'])

with col2:
    st.write("**AI Analysis**")
    st.metric("Detected Behavior", thread["ai_behavior"])
    st.write("**AI Suggestion**")
    st.write(thread["ai_suggestion"])
    # Confidence slider for demo purposes
    ai_conf = st.slider("AI Confidence", min_value=50, max_value=100, value=82)

st.markdown("---")

# Decision section
st.subheader("Decision")
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    # Approve button
    if st.button("✅ Approve Suggested Action"):
        final_action = thread["ai_suggestion"]
        status = "Approved"
        entry = {
            "thread_id": thread["thread_id"],
            "ai_suggestion": thread["ai_suggestion"],
            "final_action": final_action,
            "decision_status": status,
            "user_id": "Somesh Shukla",
            "timestamp": datetime.utcnow().isoformat(),
            "ai_confidence": ai_conf
        }
        with open(DATA_FILE, "r+") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
        st.success("Decision saved: " + final_action)

with col_b:
    # Override dropdown
    override = st.selectbox("✏ Override Action", options=ACTIONS, index=0)
    if st.button("Save Override"):
        final_action = override
        status = "Overridden"
        entry = {
            "thread_id": thread["thread_id"],
            "ai_suggestion": thread["ai_suggestion"],
            "final_action": final_action,
            "decision_status": status,
            "user_id": "Somesh Shukla",
            "timestamp": datetime.utcnow().isoformat(),
            "ai_confidence": ai_conf
        }
        with open(DATA_FILE, "r+") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
        st.success("Override saved: " + final_action)

with col_c:
    st.write("Current Decision Status: \nPending")

st.markdown("---")

# Show audit log
st.subheader("Audit Log")
with open(DATA_FILE, "r") as f:
    log = json.load(f)
if log:
    log_df = pd.DataFrame(log)
    st.dataframe(log_df)
else:
    st.write("No decisions recorded yet.")

# Analytics view (demo)
if show_analytics:
    st.markdown("---")
    st.subheader("Analytics (Demo)")
    if log:
        hist = pd.Series([e['ai_suggestion'] for e in log]).value_counts()
        st.bar_chart(hist)
    else:
        st.write("No data to show analytics.")

# Footer integration notes
st.markdown("---")
st.write("Integration notes:\n- Replace SAMPLE_THREADS with your email ingest (IMAP/webhook) and real AI classifier.\n- The AI classifier should provide behavior label and suggestion per latest reply.\n- decisions_log.json stores final decisions; use this to retrain/feedback.\n- To run: `streamlit run streamlit_email_behavior_ui.py`")
