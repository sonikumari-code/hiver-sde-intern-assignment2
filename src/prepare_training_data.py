import pandas as pd
import re

INPUT_FILE = "data/sprintcare_pairs.csv"
OUTPUT_FILE = "data/sprintcare_training.csv"


def assign_intent(text):
    text = str(text).lower()

    # Network / Connectivity
    network_words = [
        "no service", "no signal", "network", "coverage",
        "lte", "internet", "data", "slow internet", "connection",
        "connectivity", "dropped call", "drop calls", "calls dropping",
        "can't make calls", "cannot make calls", "text failing",
        "text messages", "sms", "service"
    ]

    # Billing / Payment
    billing_words = [
        "bill", "billing", "charged", "charge", "payment",
        "pay", "price", "fee", "cost", "balance", "refund",
        "overcharged", "money", "monthly"
    ]

    # Order / Upgrade / iPhone
    order_words = [
        "iphone x", "iphone 8", "iphone 7", "iphone",
        "preorder", "pre-order", "pre order", "upgrade",
        "order", "ordered", "shipping", "delivery",
        "stock", "phone upgrade"
    ]

    # Website / App
    website_words = [
        "website", "webpage", "app", "application",
        "loading", "loading screen", "crash", "crashed",
        "page", "cart", "online"
    ]

    # Cancellation / Switching
    cancellation_words = [
        "cancel", "cancellation", "switch", "switching",
        "leave sprint", "leaving sprint", "another carrier",
        "other carrier", "port", "move to", "moving to"
    ]

    # Customer Service
    service_words = [
        "customer service", "representative", "rep",
        "agent", "support", "help", "call center",
        "called", "waiting", "waited", "no one",
        "nobody", "blown off", "useless reply"
    ]

    # Device / Technical
    device_words = [
        "phone", "device", "screen", "microphone",
        "speaker", "battery", "charger", "charging",
        "activation", "activate", "broken", "defective",
        "not working", "doesn't work", "camera"
    ]

    # Check more specific categories first

    if any(word in text for word in cancellation_words):
        return "Cancellation / Switching / Retention"

    if any(word in text for word in billing_words):
        return "Billing / Payment"

    if any(word in text for word in order_words):
        return "Order / Upgrade / iPhone"

    if any(word in text for word in website_words):
        return "Website / App Issue"

    if any(word in text for word in device_words):
        return "Device / Technical Issue"

    if any(word in text for word in network_words):
        return "Network / Connectivity"

    if any(word in text for word in service_words):
        return "Customer Service Issue"

    return "General Complaint / Other"


print("=" * 60)
print("Preparing Weakly-Labeled Training Data")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Original conversations:", len(df))

df["intent"] = df["customer_message"].apply(assign_intent)

# Remove any empty messages
df = df.dropna(subset=["customer_message"])

# Remove exact duplicate customer messages
df = df.drop_duplicates(subset=["customer_message"])

df.to_csv(OUTPUT_FILE, index=False)

print("Training examples:", len(df))
print("\nIntent distribution:")
print(df["intent"].value_counts())

print("\nSaved to:", OUTPUT_FILE)
print("=" * 60)