import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Files
# --------------------------------------------------

DATA_FILE = "data/sprintcare_pairs.csv"
MODEL_FILE = "results/intent_model.joblib"
VECTORIZER_FILE = "results/tfidf_vectorizer.joblib"


# --------------------------------------------------
# Load model, vectorizer and historical data
# --------------------------------------------------

print("=" * 60)
print("HIVER - SprintCare AI Support Agent")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

df = df.dropna(
    subset=["customer_message", "brand_reply"]
).reset_index(drop=True)

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

historical_vectors = vectorizer.transform(
    df["customer_message"].astype(str)
)

print("Historical conversations:", len(df))
print("Agent loaded successfully.")


# --------------------------------------------------
# Intent classification
# --------------------------------------------------

def classify_intent(message):

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = probabilities.max()

    return prediction, confidence


# --------------------------------------------------
# Historical retrieval
# --------------------------------------------------

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
                df.iloc[index]["customer_message"],
            "brand_reply":
                df.iloc[index]["brand_reply"]
        })

    return evidence


# --------------------------------------------------
# Escalation policy
# --------------------------------------------------

def decide_escalation(intent, confidence, similarity):

    high_risk_intents = [
        "Billing / Payment",
        "Cancellation / Switching / Retention"
    ]

    # High-risk cases go to a human
    if intent in high_risk_intents:
        return (
            "Human",
            "High-risk account, billing, or cancellation issue."
        )

    # Low confidence goes to a human
    if confidence < 0.45:
        return (
            "Human",
            "Intent confidence is too low."
        )

    # Weak historical evidence goes to a human
    if similarity < 0.20:
        return (
            "Human",
            "No sufficiently similar historical resolution found."
        )

    # Otherwise allow automatic handling
    return (
        "Auto",
        "Clear intent and sufficiently similar historical evidence."
    )


# --------------------------------------------------
# Generate grounded reply
# --------------------------------------------------

def generate_reply(message):

    intent, confidence = classify_intent(message)

    evidence = retrieve_evidence(
        message,
        top_k=3
    )

    best_similarity = evidence[0]["similarity"]

    escalation, reason = decide_escalation(
        intent,
        confidence,
        best_similarity
    )

    # Use historical reply only when evidence is strong
    if best_similarity >= 0.20:

        reply = evidence[0]["brand_reply"]

    else:

        reply = (
            "Thanks for reaching out. We'd like to "
            "understand the issue better and help. "
            "Please send us a DM so we can assist you."
        )

    return {
        "intent": intent,
        "confidence": confidence,
        "escalation": escalation,
        "reason": reason,
        "reply": reply,
        "evidence": evidence
    }


# --------------------------------------------------
# Interactive agent
# --------------------------------------------------

print("\nType a customer message.")
print("Type 'exit' to stop.\n")


while True:

    message = input("Customer: ")

    if message.lower() == "exit":
        break

    result = generate_reply(message)

    print("\n" + "=" * 60)
    print("AI SUPPORT AGENT RESULT")
    print("=" * 60)

    print(
        "Intent:",
        result["intent"]
    )

    print(
        "Confidence:",
        round(result["confidence"], 3)
    )

    print(
        "Escalation:",
        result["escalation"]
    )

    print(
        "Reason:",
        result["reason"]
    )

    print("\nDraft Reply:")
    print(result["reply"])

    print("\nHistorical Evidence:")

    for i, item in enumerate(
        result["evidence"],
        start=1
    ):

        print(f"\nExample {i}")
        print(
            "Similarity:",
            round(item["similarity"], 3)
        )

        print(
            "Customer:",
            item["customer_message"]
        )

        print(
            "SprintCare:",
            item["brand_reply"]
        )

    print("\n" + "=" * 60)