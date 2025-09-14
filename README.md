# Email-Behavior-Orchestrator
The Email Behavior Orchestrator helps inbound email teams handle large volumes of replies from outbound campaigns.

An AI-powered system to classify email conversation threads into **behaviors**  
(Confirmation, Objection, Escalation, New Information, Unknown) and suggest appropriate **actions**.

---

## 🚀 Features
- Load raw threads from `Data/sample_emails.json`
- Use **Gemini (gemini-2.0-flash)** to pseudo-label data
- Train a **TF-IDF + Logistic Regression** classifier (`model.pkl`)
- Generate AI predictions saved in `predictions.json`
- Interactive **Streamlit dashboard (`app.py`)**:
  - View AI-detected behaviors
  - Approve or override suggested actions
  - Audit log of analyst decisions (`logs/decisions_log.json`)
  - Classify new test emails using trained model

---
