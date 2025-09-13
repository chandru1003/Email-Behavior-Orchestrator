import json
import re
import string
import pandas as pd
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

def clean_text(text):
    text = text.lower()
    # remove email headers/signatures
    text = re.sub(r"(from:|to:|cc:|thanks|regards|sincerely|dear|best)", " ", text)
    # remove urls
    text = re.sub(r"http\S+|www\S+", " ", text)
    # remove numbers and punctuation
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("cleaning the dataset")
df["clean_text"] = df["email_text"].apply(clean_text)

#  Vectorize + Unsupervised Clustering

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
print(" Predicting the behavior of emails...")
# Choose 4 clusters (confirmation, objection, escalation, new info)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)


#  Map clusters : behaviors → actions
cluster_to_behavior = {
    0: "Confirmation",
    1: "Objection",
    2: "Escalation",
    3: "New Information"
}

behavior_to_action = {
    "Confirmation": "Close Ticket",
    "Objection": "Request Clarification",
    "Escalation": "Forward to Manager",
    "New Information": "Update Records"
}
print("Assigning predicted behaviors and corresponding actions... ")
df["predicted_behavior"] = df["cluster"].map(cluster_to_behavior)
df["suggested_action"] = df["predicted_behavior"].map(behavior_to_action)


# Display Output
print("\n Displaying output in tabular format:\n")
print(df[["thread_id", "sender", "email_text", "predicted_behavior", "suggested_action"]])


# (Optional) Visualize clusters

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=df["cluster"], cmap="viridis")
plt.title("Email Behavior Clusters")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()
