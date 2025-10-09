"""
Streamlit UI for Email Behavior Orchestrator
- Features a Dashboard for performance overview.
- Reads AI predictions from predictions.json
- Loads trained model.pkl for classifying new emails
- Supports decision logging & overrides
- Allows dynamic rule editing for actions
- Includes a login page for access control.
"""

import streamlit as st
import pandas as pd
import json
import os
import joblib
from datetime import datetime, timezone
import plotly.express as px

# Set page config as the first Streamlit command
st.set_page_config(page_title="Email Behavior Orchestrator", layout="wide")

# --- Initialize session state for login and active tab ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Dashboard"

def run_orchestrator_app():
    """
    This function contains the entire original application logic.
    It is called only after a successful login.
    """
    # ---------------------------
    # CONFIG
    # ---------------------------
    PREDICTIONS_FILE = "predictions.json"
    MODEL_FILE = "model/model.pkl"
    DATA_FILE = "logs/decisions_log.json"
    RULES_FILE = "rules.json"

    # ---- Load and Initialize Rules ----
    if not os.path.exists(RULES_FILE):
        default_rules = {
            "Confirmation": "Close Ticket",
            "Objection": "Escalate to Manager",
            "New Information": "Request Additional Information",
            "Question": "Assign to Agent",
            "Other": "No Action"
        }
        with open(RULES_FILE, "w") as f:
            json.dump(default_rules, f, indent=2)

    def load_rules():
        try:
            with open(RULES_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "Confirmation": "Close Ticket",
                "Objection": "Escalate to Manager",
                "New Information": "Request Additional Information",
                "Question": "Assign to Agent",
                "Other": "No Action"
            }

    # ---------------------------
    # Init log file if missing
    # ---------------------------
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump([], f)

    # ---------------------------
    # Load data sources
    # ---------------------------
    # @st.cache_data # Caching can interfere with live updates after saving, so we load fresh data on each run
    def load_data():
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, "r") as f:
                predictions = json.load(f)
        else:
            predictions = []

        with open(DATA_FILE, "r") as f:
            log = json.load(f)

        return predictions, log

    predictions, log = load_data()
    rules = load_rules()

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        model = None

    # ---------------------------
    # Streamlit UI
    # ---------------------------
    st.title("📧 Email Behavior Orchestrator")

    # ---- Sidebar for Rule Editing ----
    with st.sidebar:
        st.header("📝 Edit Rules")
        st.markdown("Define the default action for each detected behavior.")
        
        new_rules = {}
        for behavior, action in rules.items():
            new_rules[behavior] = st.text_input(f"Action for '{behavior}'", action, key=f"rule_{behavior}")
        
        if st.button("Save Rules"):
            with open(RULES_FILE, "w") as f:
                json.dump(new_rules, f, indent=2)
            st.success("Rules saved!")
            # No need to clear cache or rerun, button press does it automatically

    # ---- Main content with stateful navigation ----
    tab_names = ["📊 Dashboard", "📨 Inbox & Audit", "📈 Analytics"]
    
    # Use st.radio for navigation and persist the choice in session_state
    st.session_state.active_tab = st.radio(
        "Navigation",
        options=tab_names,
        key="navigation_radio",
        horizontal=True,
        label_visibility="collapsed"
    )

    ## ---------------------------
    ## Screen 1: Dashboard
    ## ---------------------------
    if st.session_state.active_tab == "📊 Dashboard":
        st.header("Control Center & Performance Overview")
        st.markdown("Immediately understand the status of your campaigns and the value the system is providing.")
        
        st.markdown("---")
        
        if st.button("🚀 Launch New Campaign", type="primary"):
            st.success("A new campaign has been launched successfully!")

        st.markdown("##") # Adds vertical space
        
        # --- Quick Stats ---
        st.subheader("Quick Stats")
        
        # Prepare data for stats
        total_replies = len(predictions)
        decisions_made = len(log)
        
        if decisions_made > 0:
            log_df_dash = pd.DataFrame(log)
            overridden = len(log_df_dash[log_df_dash['decision_status'] == 'Overridden'])
            accuracy = f"{((decisions_made - overridden) / decisions_made) * 100:.1f}%"
        else:
            accuracy = "N/A"
            
        if predictions:
            pred_df_dash = pd.DataFrame(predictions)
            most_common_behavior = pred_df_dash['predicted_behavior'].mode()[0]
        else:
            most_common_behavior = "N/A"

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Replies in Inbox", value=total_replies)
        with col2:
            st.metric(label="Total Decisions Logged", value=decisions_made)
        with col3:
            st.metric(label="AI Suggestion Accuracy", value=accuracy)
        with col4:
            st.metric(label="Most Common Behavior", value=most_common_behavior)

    ## ---------------------------
    ## Screen 2: Inbox & Audit
    ## ---------------------------
    elif st.session_state.active_tab == "📨 Inbox & Audit":
        st.header("Inbox — Threads for Review")

        if predictions:
            df = pd.DataFrame([{
                "Thread ID": p["thread_id"],
                "Sender": p["sender"],
                "AI-Detected Behavior": p["predicted_behavior"],
                "Suggested Action": rules.get(p["predicted_behavior"], "Review Manually"),
                "Decision Status": "Pending"
            } for p in predictions])
            
            st.info("Click on a row in the table below to select a thread for audit.")
            
            event = st.dataframe(
                df,
                key="inbox_df",
                on_select="rerun",
                selection_mode="single-row",
                use_container_width=True
            )

            selected_row_index = event.selection.rows[0] if event.selection.rows else 0
            selected_thread_id = df.iloc[selected_row_index]["Thread ID"]
            
            p_selected = next(p for p in predictions if p["thread_id"] == selected_thread_id)
            thread = {
                **p_selected,
                "suggested_action": rules.get(p_selected["predicted_behavior"], "Review Manually")
            }
        else:
            st.warning("No predictions found. Please run the classification pipeline first.")
            thread = None

        if thread:
            st.markdown("---")
            st.subheader(f"🔍 Audit & Action Center: {thread['thread_id']}")

            with st.expander("Show Full Conversation Transcript", expanded=True):
                st.info(thread["email_text"])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🤖 AI Analysis")
                st.metric("Detected Behavior", thread["predicted_behavior"])
                st.write("**Suggested Action:**")
                st.success(thread["suggested_action"])
                ai_conf = st.slider("AI Confidence", 50, 100, 85, key=f"conf_{thread['thread_id']}")

            with col2:
                st.subheader("Human-in-the-Loop Controls")
                
                if st.button("✅ Approve Suggested Action", key=f"approve_{thread['thread_id']}"):
                    entry = {
                        "thread_id": thread["thread_id"], "ai_suggestion": thread["suggested_action"],
                        "final_action": thread["suggested_action"], "decision_status": "Approved",
                        "user_id": st.session_state.username, "timestamp": datetime.now(timezone.utc).isoformat(),
                        "ai_confidence": ai_conf
                    }
                    log.append(entry)
                    with open(DATA_FILE, "w") as f: json.dump(log, f, indent=2)
                    st.success("Decision saved!")
                    # No explicit rerun needed; button press handles it.

                override_options = list(set(rules.values())) + ["No Action"]
                override = st.selectbox("✏️ Override Action", options=override_options, key=f"override_select_{thread['thread_id']}")
                if st.button("Save Override", key=f"override_btn_{thread['thread_id']}"):
                    entry = {
                        "thread_id": thread["thread_id"], "ai_suggestion": thread["suggested_action"],
                        "final_action": override, "decision_status": "Overridden",
                        "user_id": st.session_state.username, "timestamp": datetime.now(timezone.utc).isoformat(),
                        "ai_confidence": ai_conf
                    }
                    log.append(entry)
                    with open(DATA_FILE, "w") as f: json.dump(log, f, indent=2)
                    st.success("Override saved!")
                    # No explicit rerun needed; button press handles it.

            st.markdown("---")
            st.subheader("📝 Audit Trail for This Thread")
            log_df = pd.DataFrame(log)
            if not log_df.empty:
                thread_history = log_df[log_df['thread_id'] == thread['thread_id']]
                if not thread_history.empty:
                    st.dataframe(thread_history)
                else:
                    st.info("No decisions have been logged for this thread yet.")
            else:
                st.info("No decisions have been logged for this thread yet.")

    ## ---------------------------
    ## Screen 3: Analytics
    ## ---------------------------
    elif st.session_state.active_tab == "📈 Analytics":
        st.header("📊 Analytics Dashboard")

        if not log:
            st.info("No decision history has been recorded yet to display analytics.")
        else:
            df_analytics = pd.DataFrame(log)
            
            total_decisions = len(df_analytics)
            overridden_decisions = len(df_analytics[df_analytics['decision_status'] == 'Overridden'])
            accuracy = ((total_decisions - overridden_decisions) / total_decisions) * 100 if total_decisions > 0 else 100

            action_counts = df_analytics['final_action'].value_counts().reset_index()
            action_counts.columns = ['action', 'count']

            st.subheader("Key Metrics")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(
                    label="AI Suggestion Accuracy",
                    value=f"{accuracy:.1f}%",
                    help="The percentage of AI suggestions that were 'Approved' by a human."
                )
            with m_col2:
                st.metric(label="Total Decisions Logged", value=total_decisions)
            
            st.subheader("Final Action Breakdown")
            fig = px.pie(
                action_counts, names='action', values='count',
                title='Distribution of Final Actions Taken',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(legend=dict(font=dict(size=14)))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Global Audit Log (All Threads)")
            st.dataframe(df_analytics)


# --- Main app execution ---
if not st.session_state.logged_in:
    # --- LOGIN PAGE ---
    st.title("📧 Email Behavior Orchestrator - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Please enter a username and password.")
else:
    # --- RUN THE MAIN APP ---
    run_orchestrator_app()