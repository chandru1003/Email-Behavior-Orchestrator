import json
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
print("Reading the sample Data set")

with open("Data\sample_emails.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["email_text"] = df["messages"].apply(lambda msgs: msgs[-1] if isinstance(msgs, list) and msgs else "")

# Clean email text
# NLP Preprocessing with spaCy

nlp = spacy.load("en_core_web_sm")

CUSTOM_STOPWORDS = {"thanks", "regards", "sincerely", "dear", "best", "from", "to", "cc", "subject"}

def clean_text_spacy(text: str) -> str:
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num:
            continue
        if token.text in CUSTOM_STOPWORDS:
            continue
        tokens.append(token.lemma_)  # use lemmatization
    return " ".join(tokens)

df["clean_text"] = df["email_text"].apply(clean_text_spacy)


#  Vectorize + Unsupervised Clustering

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
print(" Predicting the behavior of emails...")
# Choose 4 clusters (confirmation, objection, escalation, new info)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# ---------------------------
# 4. Load cluster-to-action mapping from JSON
# ---------------------------
# Example mapping file (actions.json):
# {
#   "cluster_to_behavior": {
#     "0": "Confirmation",
#     "1": "Objection",
#     "2": "Escalation",
#     "3": "New Information"
#   },
#   "behavior_to_action": {
#     "Confirmation": "Close Ticket",
#     "Objection": "Request Clarification",
#     "Escalation": "Forward to Manager",
#     "New Information": "Update Records"
#   }
# }

with open("action.json", "r") as f:
    mapping = json.load(f)

cluster_to_behavior = {int(k): v for k, v in mapping["cluster_to_behavior"].items()}
behavior_to_action = mapping["behavior_to_action"]

df["predicted_behavior"] = df["cluster"].map(cluster_to_behavior)
df["suggested_action"] = df["predicted_behavior"].map(behavior_to_action)

# Display Output
print("\n Displaying output in tabular format:\n")
print(df[["thread_id", "sender", "email_text", "predicted_behavior", "suggested_action"]].head(10))
