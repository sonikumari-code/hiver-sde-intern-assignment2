import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity


GOLDEN_FILE = "data/golden_200_reviewed.csv"
HISTORY_FILE = "data/sprintcare_history_clean.csv"

MODEL_FILE = "results/clean_intent_model.joblib"
VECTORIZER_FILE = "results/clean_tfidf_vectorizer.joblib"

OUTPUT_FILE = "results/clean_support_agent_evaluation.csv"


print("=" * 60)
print("CLEAN HIVER SUPPORT AGENT EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

golden = pd.read_csv(GOLDEN_FILE)

history = pd.read_csv(HISTORY_FILE)

history = history.dropna(
    subset=["customer_message", "brand_reply"]
).reset_index(drop=True)

golden = golden.dropna(
    subset=["customer_message", "intent", "escalation"]
).reset_index(drop=True)


print("Golden test examples:", len(golden))
print("Clean historical examples:", len(history))


# ---------------------------------------------------------
# Check leakage
# ---------------------------------------------------------

golden_ids = set(
    golden["customer_tweet_id"].astype(str)
)

history_ids = set(
    history["customer_tweet_id"].astype(str)
)

id_overlap = golden_ids.intersection(history_ids)

print("Golden ID overlap:", len(id_overlap))


# ---------------------------------------------------------
# Load clean model
# ---------------------------------------------------------

model = joblib.load(MODEL_FILE)

vectorizer = joblib.load(VECTORIZER_FILE)


# ---------------------------------------------------------
# Historical vectors
# ---------------------------------------------------------

historical_vectors = vectorizer.transform(
    history["customer_message"].astype(str)
)


# ---------------------------------------------------------
# Intent classification
# ---------------------------------------------------------

def classify_intent(message):

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = probabilities.max()

    return prediction, confidence


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_evidence(message, top_k=3):

    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        historical_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    evidence = []

    for index in top_indices:

        evidence.append({
            "similarity": float(similarities[index]),
            "customer_message":
                history.iloc[index]["customer_message"],
            "brand_reply":
                history.iloc[index]["brand_reply"]
        })

    return evidence


# ---------------------------------------------------------
# Escalation
# ---------------------------------------------------------

def decide_escalation(intent, confidence, similarity):

    high_risk_intents = [
        "Billing / Payment",
        "Cancellation / Switching / Retention"
    ]

    if intent in high_risk_intents:
        return "Human"

    if confidence < 0.45:
        return "Human"

    if similarity < 0.20:
        return "Human"

    return "Auto"


# ---------------------------------------------------------
# Evaluate golden examples
# ---------------------------------------------------------

predicted_intents = []
predicted_escalations = []

confidences = []
similarities = []
replies = []


for message in golden["customer_message"].astype(str):

    intent, confidence = classify_intent(message)

    evidence = retrieve_evidence(
        message,
        top_k=3
    )

    best_similarity = evidence[0]["similarity"]

    escalation = decide_escalation(
        intent,
        confidence,
        best_similarity
    )

    if best_similarity >= 0.20:

        reply = evidence[0]["brand_reply"]

    else:

        reply = (
            "Thanks for reaching out. "
            "We'd like to understand the issue better "
            "and help. Please send us a DM so we can assist you."
        )

    predicted_intents.append(intent)
    predicted_escalations.append(escalation)

    confidences.append(confidence)
    similarities.append(best_similarity)

    replies.append(reply)


# ---------------------------------------------------------
# Store predictions
# ---------------------------------------------------------

golden["predicted_intent"] = predicted_intents

golden["intent_confidence"] = confidences

golden["predicted_escalation"] = predicted_escalations

golden["best_similarity"] = similarities

golden["draft_reply"] = replies


# ---------------------------------------------------------
# Intent metrics
# ---------------------------------------------------------

intent_accuracy = accuracy_score(
    golden["intent"],
    golden["predicted_intent"]
)

intent_macro_f1 = f1_score(
    golden["intent"],
    golden["predicted_intent"],
    average="macro"
)


# ---------------------------------------------------------
# Escalation metrics
# ---------------------------------------------------------

escalation_accuracy = accuracy_score(
    golden["escalation"],
    golden["predicted_escalation"]
)

escalation_macro_f1 = f1_score(
    golden["escalation"],
    golden["predicted_escalation"],
    average="macro"
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("INTENT RESULTS")
print("=" * 60)

print(
    "Intent Accuracy:",
    round(intent_accuracy, 4)
)

print(
    "Intent Accuracy %:",
    round(intent_accuracy * 100, 2)
)

print(
    "Intent Macro F1:",
    round(intent_macro_f1, 4)
)


print("\nIntent Classification Report:")

print(
    classification_report(
        golden["intent"],
        golden["predicted_intent"],
        zero_division=0
    )
)


print("\n" + "=" * 60)
print("ESCALATION RESULTS")
print("=" * 60)

print(
    "Escalation Accuracy:",
    round(escalation_accuracy, 4)
)

print(
    "Escalation Accuracy %:",
    round(escalation_accuracy * 100, 2)
)

print(
    "Escalation Macro F1:",
    round(escalation_macro_f1, 4)
)


print("\nEscalation Classification Report:")

print(
    classification_report(
        golden["escalation"],
        golden["predicted_escalation"],
        zero_division=0
    )
)


# ---------------------------------------------------------
# Retrieval statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RETRIEVAL STATISTICS")
print("=" * 60)

print(
    "Average similarity:",
    round(
        golden["best_similarity"].mean(),
        4
    )
)

print(
    "Median similarity:",
    round(
        golden["best_similarity"].median(),
        4
    )
)

print(
    "Examples with similarity >= 0.20:",
    int(
        (golden["best_similarity"] >= 0.20).sum()
    )
)

print(
    "Examples with similarity < 0.20:",
    int(
        (golden["best_similarity"] < 0.20).sum()
    )
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_columns = [
    "customer_tweet_id",
    "customer_message",
    "intent",
    "predicted_intent",
    "intent_confidence",
    "escalation",
    "predicted_escalation",
    "best_similarity",
    "draft_reply"
]

golden[output_columns].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("Evaluation:", OUTPUT_FILE)

print("=" * 60)