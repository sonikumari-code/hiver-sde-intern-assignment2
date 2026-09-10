import pandas as pd

INPUT_FILE = "data/sprintcare_history_clean.csv"
OUTPUT_FILE = "data/sprintcare_training_clean.csv"


def assign_weak_label(text):
    text = str(text).lower()

    # Network / Connectivity
    network_keywords = [
        "network", "signal", "service", "lte", "4g", "3g",
        "internet", "data", "wifi", "wi-fi", "connection",
        "connect", "coverage", "slow", "disconnect",
        "dropped call", "call drop", "no service", "not working"
    ]

    # Billing / Payment
    billing_keywords = [
        "bill", "billing", "charge", "charged", "payment",
        "pay", "paid", "fee", "price", "cost", "refund",
        "money", "balance", "credit", "debit", "invoice"
    ]

    # Device / Technical Issue
    device_keywords = [
        "phone", "device", "iphone", "android", "screen",
        "sim", "sim card", "activation", "activate",
        "broken", "repair", "battery", "camera",
        "software", "hardware"
    ]

    # Order / Upgrade / iPhone
    order_keywords = [
        "upgrade", "preorder", "pre-order", "order",
        "shipping", "ship", "delivery", "iphone x",
        "iphone 8", "iphone 7", "new iphone"
    ]

    # Website / App Issue
    website_keywords = [
        "website", "web site", "app", "application",
        "login", "log in", "logged in", "loading",
        "crash", "browser", "cart", "checkout",
        "online"
    ]

    # Customer Service Issue
    service_keywords = [
        "customer service", "support", "agent", "representative",
        "rep", "help", "help me", "waiting", "waited",
        "call", "called", "dm", "response", "respond",
        "nobody", "no one"
    ]

    # Cancellation / Switching / Retention
    cancellation_keywords = [
        "cancel", "cancellation", "leave sprint",
        "leaving sprint", "switch", "switching",
        "another carrier", "other carrier",
        "verizon", "at&t", "att", "t-mobile",
        "tmobile", "port", "ported", "terminate"
    ]

    # Apply specific categories first
    if any(keyword in text for keyword in cancellation_keywords):
        return "Cancellation / Switching / Retention"

    if any(keyword in text for keyword in billing_keywords):
        return "Billing / Payment"

    if any(keyword in text for keyword in order_keywords):
        return "Order / Upgrade / iPhone"

    if any(keyword in text for keyword in website_keywords):
        return "Website / App Issue"

    if any(keyword in text for keyword in device_keywords):
        return "Device / Technical Issue"

    if any(keyword in text for keyword in network_keywords):
        return "Network / Connectivity"

    if any(keyword in text for keyword in service_keywords):
        return "Customer Service Issue"

    return "General Complaint / Other"


print("=" * 60)
print("CREATING CLEAN WEAK-LABEL TRAINING DATA")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Input rows:", len(df))

# Remove rows without required text
df = df.dropna(
    subset=["customer_message", "brand_reply"]
).copy()

# Remove exact duplicate customer messages
df["customer_message"] = (
    df["customer_message"].astype(str).str.strip()
)

df = df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

# Create weak labels
df["intent"] = df["customer_message"].apply(assign_weak_label)

# Save
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Training rows:", len(df))

print("\nWeak-label distribution:")
print(df["intent"].value_counts())

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 60)