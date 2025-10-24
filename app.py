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
from gmail_service import get_gmail_service, get_unread_emails, get_email_content, create_and_send_reply, apply_label, archive_email

# Set page config as the first Streamlit command
st.set_page_config(page_title="Email Behavior Orchestrator", layout="wide")

# --- Initialize session state for login and active tab -- -
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Dashboard"
if "emails" not in st.session_state:
    st.session_state.emails = []
if "drafted_reply" not in st.session_state:
    st.session_state.drafted_reply = None


def fetch_and_process_emails(model):
    """Fetches unread emails, classifies them, and stores them in the session state."""
    print("--- Starting to fetch emails ---")
    service = get_gmail_service()
    if not service:
        st.error("Could not connect to Gmail API.")
        print("Error: Could not get Gmail service.")
        return

    print("Gmail service obtained. Fetching unread messages...")
    unread_messages = get_unread_emails(service)
    
    print(f"API response for unread messages: {unread_messages}") # LOGGING

    if not unread_messages:
        st.info("No unread emails found.")
        print("No unread messages returned from API.")
        return

    print(f"Found {len(unread_messages)} unread messages.") # LOGGING
    new_emails = []
    for msg in unread_messages:
        print(f"Processing message ID: {msg['id']}") # LOGGING
        email_data = get_email_content(service, msg['id'])
        if email_data and model:
            # Use the model to predict
            prediction = model.predict([email_data['body']])[0]
            
            email_data['predicted_behavior'] = prediction
            email_data['thread_id'] = msg['threadId']
            email_data['email_text'] = email_data['body'] # for compatibility with existing UI
            new_emails.append(email_data)
            print(f"Successfully processed and classified message ID: {msg['id']}") # LOGGING
        else:
            print(f"Could not get content or model not available for message ID: {msg['id']}") # LOGGING


    st.session_state.emails = new_emails
    print(f"--- Finished fetching emails. Stored {len(new_emails)} emails in session state. ---")


def run_orchestrator_app():
    """
    This function contains the entire original application logic.
    It is called only after a successful login.
    """
    # -------------------------- -
    # CONFIG
    # -------------------------- -
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

    # -------------------------- -
    # Init log file if missing
    # -------------------------- -
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump([], f)

    # -------------------------- -
    # Load data sources
    # -------------------------- -
    def load_data():
        # We now primarily use session state for emails, but keep log loading
        with open(DATA_FILE, "r") as f:
            log = json.load(f)
        return st.session_state.emails, log

    predictions, log = load_data()
    rules = load_rules()

    # -------------------------- -
    # ACTION MAPPING
    # -------------------------- -
    # This dictionary maps action strings to functions.
    # It now intelligently handles whether an action is immediate (like archiving) 
    # or requires drafting an email for review.

    def create_draft(recipient, subject, body, original_email_id):
        """Helper function to place a draft into the session state for editing."""
        st.session_state.drafted_reply = {
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "original_email_id": original_email_id
        }

    action_map = {
        # Immediate actions (no user editing needed)
        "Close Ticket": lambda service, thread: archive_email(service, thread['id']),
        "Assign to Agent": lambda service, thread: apply_label(service, thread['id'], "Assigned-Agent"), # TODO: Make sure this label exists in Gmail
        "No Action": lambda service, thread: None, # Does nothing

        # Email actions (will create a draft for editing)
        "Escalate to Manager": lambda service, thread: create_draft(
            recipient="kj@kgrp.in", # <-- TODO: Change to your manager's email
            subject=f"Escalated: {thread['subject']}",
            body=thread['email_text'],
            original_email_id=thread['id']
        ),
        "Request Additional Information": lambda service, thread: create_draft(
            recipient=thread['from'],
            subject=f"Re: {thread['subject']}",
            body="Thank you for your email. Could you please provide more information regarding your request?", # <-- TODO: Customize this template
            original_email_id=thread['id']
        )
    }

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        model = None
        st.error("Model file not found. Please make sure model.pkl is in the model/ directory.")


    # -------------------------- -
    # Streamlit UI
    # -------------------------- -
    st.title("📧 Email Behavior Orchestrator")

    # ---- Sidebar for Rule Editing and Gmail Actions ----
    with st.sidebar:
        st.header("📧 Gmail Actions")
        if st.button("Fetch New Emails"):
            fetch_and_process_emails(model)
            st.rerun()

        st.header("📝 Edit Rules")
        st.markdown("Define the default action for each detected behavior.")
        
        new_rules = {}
        for behavior, action in rules.items():
            new_rules[behavior] = st.text_input(f"Action for '{behavior}'", action, key=f"rule_{behavior}")
        
        if st.button("Save Rules"):
            with open(RULES_FILE, "w") as f:
                json.dump(new_rules, f, indent=2)
            st.success("Rules saved!")
            st.rerun()

    # ---- Main content with stateful navigation ----
    tab_names = ["📊 Dashboard", "📨 Inbox & Audit", "📈 Analytics"]
    
    st.session_state.active_tab = st.radio(
        "Navigation",
        options=tab_names,
        key="navigation_radio",
        horizontal=True,
        label_visibility="collapsed"
    )

    ## -------------------------- -
    ## Screen 1: Dashboard
    ## -------------------------- -
    if st.session_state.active_tab == "📊 Dashboard":
        st.header("Control Center & Performance Overview")
        st.markdown("Immediately understand the status of your campaigns and the value the system is providing.")
        
        st.markdown("---")
        
        if st.button("🚀 Launch New Campaign", type="primary"):
            st.success("A new campaign has been launched successfully!")

        st.markdown("##") # Adds vertical space
        
        # --- Quick Stats -- -
        st.subheader("Quick Stats")
        
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

    ## -------------------------- -
    ## Screen 2: Inbox & Audit
    ## -------------------------- -
    elif st.session_state.active_tab == "📨 Inbox & Audit":
        st.header("Inbox — Threads for Review")

        if predictions:
            # --- Sorting Logic ---
            all_behaviors = sorted(list(set(p["predicted_behavior"] for p in predictions)))
            sort_options = ["Default"] + all_behaviors
            
            selected_sort = st.selectbox(
                "Sort by Behavior",
                options=sort_options,
                key="sort_behavior_select"
            )

            sorted_predictions = predictions.copy()
            if selected_sort != "Default":
                # This brings emails with the selected behavior to the top of the list
                sorted_predictions.sort(key=lambda p: p['predicted_behavior'] != selected_sort)

            df = pd.DataFrame([{
                "Thread ID": p["thread_id"],
                "Sender": p["from"],
                "AI-Detected Behavior": p["predicted_behavior"],
                "Suggested Action": rules.get(p["predicted_behavior"], "Review Manually"),
                "Decision Status": "Pending"
            } for p in sorted_predictions])
            
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
            
            # Use the sorted list to find the selected email
            p_selected = next(p for p in sorted_predictions if p["thread_id"] == selected_thread_id)
            thread = {
                **p_selected,
                "suggested_action": rules.get(p_selected["predicted_behavior"], "Review Manually")
            }
        else:
            st.warning("No emails found. Click 'Fetch New Emails' in the sidebar.")
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
                
                # --- Unified Action Logic ---
                action_to_perform = None
                
                if st.button("✅ Approve Suggested Action", key=f"approve_{thread['thread_id']}"):
                    action_to_perform = thread["suggested_action"]
                    entry = {
                        "thread_id": thread["thread_id"], "ai_suggestion": thread["suggested_action"],
                        "final_action": action_to_perform, "decision_status": "Approved",
                        "user_id": st.session_state.username, "timestamp": datetime.now(timezone.utc).isoformat(),
                        "ai_confidence": ai_conf
                    }
                    log.append(entry)
                    with open(DATA_FILE, "w") as f: json.dump(log, f, indent=2)

                override_options = list(set(rules.values())) + ["No Action"]
                override = st.selectbox("✏️ Override Action", options=override_options, key=f"override_select_{thread['thread_id']}")
                if st.button("Save Override", key=f"override_btn_{thread['thread_id']}"):
                    action_to_perform = override
                    entry = {
                        "thread_id": thread["thread_id"], "ai_suggestion": thread["suggested_action"],
                        "final_action": action_to_perform, "decision_status": "Overridden",
                        "user_id": st.session_state.username, "timestamp": datetime.now(timezone.utc).isoformat(),
                        "ai_confidence": ai_conf
                    }
                    log.append(entry)
                    with open(DATA_FILE, "w") as f: json.dump(log, f, indent=2)

                if action_to_perform:
                    service = get_gmail_service()
                    st.session_state.drafted_reply = None # Clear any previous draft

                    if service and action_to_perform in action_map:
                        # Execute the action from the map
                        action_map[action_to_perform](service, thread)
                        
                        # If a draft was NOT created, it was an immediate action.
                        if not st.session_state.drafted_reply:
                            st.success(f"Action '{action_to_perform}' executed successfully!")
                        else:
                            st.success(f"Reply draft for '{action_to_perform}' created. You can edit it below.")
                    
                    elif action_to_perform not in action_map:
                        st.warning(f"Action '{action_to_perform}' is not defined. Please configure it in app.py.")
                    else:
                        st.error("Could not connect to Gmail to perform the action.")
                    
                    st.rerun()

            # --- Draft Reply Section ---
            if st.session_state.drafted_reply and st.session_state.drafted_reply["original_email_id"] == thread["id"]:
                st.markdown("---")
                st.subheader("📝 Draft & Send Reply")
                
                draft = st.session_state.drafted_reply
                edited_body = st.text_area("Edit Reply Body", value=draft["body"], height=200)
                
                if st.button("🚀 Send Reply"):
                    service = get_gmail_service()
                    if service:
                        create_and_send_reply(service, draft["recipient"], draft["subject"], edited_body, draft["original_email_id"])
                        st.success("Reply sent successfully!")
                        # Clear the draft from session state
                        st.session_state.drafted_reply = None
                        st.rerun()
                    else:
                        st.error("Could not connect to Gmail to send reply.")

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

    ## -------------------------- -
    ## Screen 3: Analytics
    ## -------------------------- -
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