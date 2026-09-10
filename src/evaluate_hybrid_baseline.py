import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

INPUT_FILE = "data/golden_200_reviewed.csv"

df = pd.read_csv(INPUT_FILE)


def hybrid_intent(text):
    text = str(text).lower()

    # 1. Billing / Payment
    if any(word in text for word in [
        "bill", "billing", "charged", "charge",
        "payment", "balance", "fee", "overcharge"
    ]):
        return "Billing / Payment"

    # 2. Cancellation / Switching / Retention
    if any(word in text for word in [
        "cancel", "cancellation", "switch",
        "switching", "leave sprint", "leaving sprint",
        "verizon", "att"
    ]):
        return "Cancellation / Switching / Retention"

    # 3. Order / Upgrade / iPhone
    if any(word in text for word in [
        "iphone", "preorder", "pre-order",
        "upgrade", "order", "shipping",
        "delivery"
    ]):
        return "Order / Upgrade / iPhone"

    # 4. Website / App Issue
    if any(word in text for word in [
        "website", "webpage", "site", "app",
        "cart", "loading screen", "crash"
    ]):
        return "Website / App Issue"

    # 5. Customer Service Issue
    if any(word in text for word in [
        "customer service", "customer support",
        "agent", "representative", "rep",
        "call dropped", "no one can help",
        "support"
    ]):
        return "Customer Service Issue"

    # 6. Device / Technical Issue
    if any(word in text for word in [
        "phone", "device", "screen",
        "activation", "activate",
        "sim", "defective"
    ]):
        return "Device / Technical Issue"

    # 7. Network / Connectivity
    if any(word in text for word in [
        "service", "signal", "network",
        "internet", "lte", "3g", "4g",
        "slow", "no service", "dropped calls",
        "calls dropping", "coverage",
        "text failing"
    ]):
        return "Network / Connectivity"

    # 8. General Complaint / Other
    return "General Complaint / Other"


df["prediction"] = df["customer_message"].apply(hybrid_intent)

accuracy = accuracy_score(
    df["intent"],
    df["prediction"]
)

macro_f1 = f1_score(
    df["intent"],
    df["prediction"],
    average="macro"
)

print("=" * 60)
print("HYBRID BASELINE")
print("=" * 60)

print("Accuracy:", round(accuracy, 4))
print("Accuracy %:", round(accuracy * 100, 2))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        df["intent"],
        df["prediction"],
        zero_division=0
    )
)

df[
    [
        "customer_tweet_id",
        "customer_message",
        "intent",
        "prediction"
    ]
].to_csv(
    "results/hybrid_baseline_predictions.csv",
    index=False
)

print("\nSaved:")
print("results/hybrid_baseline_predictions.csv")