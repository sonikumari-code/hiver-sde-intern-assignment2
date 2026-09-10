import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, classification_report

HISTORY_FILE = "data/sprintcare_pairs.csv"
GOLDEN_FILE = "data/golden_200_reviewed.csv"

history = pd.read_csv(HISTORY_FILE)
golden = pd.read_csv(GOLDEN_FILE)

history = history.dropna(
    subset=["customer_message", "brand_reply"]
).reset_index(drop=True)

golden = golden.dropna(
    subset=["customer_message", "intent"]
).reset_index(drop=True)

print("=" * 60)
print("RETRIEVAL-BASED INTENT CLASSIFIER")
print("=" * 60)

print("Historical examples:", len(history))
print("Golden examples:", len(golden))

# Create TF-IDF representation of historical customer messages
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english",
    sublinear_tf=True
)

history_vectors = vectorizer.fit_transform(
    history["customer_message"].astype(str)
)

predictions = []
similarities_list = []

for message in golden["customer_message"].astype(str):

    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        history_vectors
    )[0]

    # Take top 5 historical examples
    top_indices = similarities.argsort()[-5:][::-1]

    top_labels = []

    for index in top_indices:
        historical_text = history.iloc[index]["customer_message"].lower()

        # Simple intent inference from historical conversation text
        if any(word in historical_text for word in [
            "bill", "billing", "charged", "charge",
            "payment", "balance", "fee", "overcharge"
        ]):
            label = "Billing / Payment"

        elif any(word in historical_text for word in [
            "cancel", "cancellation", "switch",
            "leaving sprint", "leave sprint",
            "verizon", "att"
        ]):
            label = "Cancellation / Switching / Retention"

        elif any(word in historical_text for word in [
            "iphone", "preorder", "pre-order",
            "upgrade", "order", "shipping",
            "delivery"
        ]):
            label = "Order / Upgrade / iPhone"

        elif any(word in historical_text for word in [
            "website", "webpage", "site", "app",
            "cart", "loading", "crash"
        ]):
            label = "Website / App Issue"

        elif any(word in historical_text for word in [
            "customer service", "customer support",
            "agent", "representative", "rep",
            "support", "call dropped"
        ]):
            label = "Customer Service Issue"

        elif any(word in historical_text for word in [
            "phone", "device", "screen",
            "activation", "activate",
            "sim", "defective"
        ]):
            label = "Device / Technical Issue"

        elif any(word in historical_text for word in [
            "service", "signal", "network",
            "internet", "lte", "3g", "4g",
            "slow", "no service", "coverage"
        ]):
            label = "Network / Connectivity"

        else:
            label = "General Complaint / Other"

        top_labels.append(label)

    # Majority vote among top 5 retrieved examples
    prediction = pd.Series(top_labels).value_counts().index[0]

    predictions.append(prediction)
    similarities_list.append(float(similarities[top_indices[0]]))


golden["prediction"] = predictions
golden["best_similarity"] = similarities_list

accuracy = accuracy_score(
    golden["intent"],
    golden["prediction"]
)

macro_f1 = f1_score(
    golden["intent"],
    golden["prediction"],
    average="macro"
)

print("\n" + "=" * 60)
print("RETRIEVAL RESULTS")
print("=" * 60)

print("Accuracy:", round(accuracy, 4))
print("Accuracy %:", round(accuracy * 100, 2))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        golden["intent"],
        golden["prediction"],
        zero_division=0
    )
)

golden[
    [
        "customer_tweet_id",
        "customer_message",
        "intent",
        "prediction",
        "best_similarity"
    ]
].to_csv(
    "results/retrieval_intent_predictions.csv",
    index=False
)

print("\nSaved:")
print("results/retrieval_intent_predictions.csv")