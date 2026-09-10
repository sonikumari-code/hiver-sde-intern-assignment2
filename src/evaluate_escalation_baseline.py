import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

INPUT_FILE = "data/golden_200_reviewed.csv"

df = pd.read_csv(INPUT_FILE)


def predict_escalation(row):
    text = str(row["customer_message"]).lower()
    intent = str(row["intent"])

    # High-risk intents → Human
    if intent in [
        "Billing / Payment",
        "Cancellation / Switching / Retention"
    ]:
        return "Human"

    # Strong customer-service frustration → Human
    if any(word in text for word in [
        "refund",
        "fraud",
        "scam",
        "stolen",
        "complaint",
        "manager",
        "supervisor",
        "lawsuit",
        "legal",
        "angry",
        "furious",
        "worst",
        "terrible",
        "unacceptable",
        "no one can help"
    ]):
        return "Human"

    # Otherwise → Auto
    return "Auto"


df["prediction"] = df.apply(
    predict_escalation,
    axis=1
)

accuracy = accuracy_score(
    df["escalation"],
    df["prediction"]
)

macro_f1 = f1_score(
    df["escalation"],
    df["prediction"],
    average="macro"
)

print("=" * 60)
print("ESCALATION BASELINE")
print("=" * 60)

print("Accuracy:", round(accuracy, 4))
print("Accuracy %:", round(accuracy * 100, 2))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        df["escalation"],
        df["prediction"],
        zero_division=0
    )
)

print("\nActual distribution:")
print(df["escalation"].value_counts())

print("\nPredicted distribution:")
print(df["prediction"].value_counts())

df[
    [
        "customer_tweet_id",
        "customer_message",
        "intent",
        "escalation",
        "prediction"
    ]
].to_csv(
    "results/escalation_baseline_predictions.csv",
    index=False
)

print("\nSaved:")
print("results/escalation_baseline_predictions.csv")