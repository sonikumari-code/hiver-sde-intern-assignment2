import pandas as pd

INPUT_FILE = "data/golden_200_labeled.csv"
OUTPUT_FILE = "data/golden_200_reviewed.csv"

df = pd.read_csv(INPUT_FILE)

corrections = {
    # First 22 corrections
    623358: "Network / Connectivity",
    44668: "Device / Technical Issue",
    967527: "Cancellation / Switching / Retention",
    928827: "Network / Connectivity",
    2702871: "Order / Upgrade / iPhone",
    2161328: "Billing / Payment",
    943214: "Network / Connectivity",
    1032968: "Cancellation / Switching / Retention",
    2042360: "Network / Connectivity",
    1318628: "Order / Upgrade / iPhone",
    1033525: "Customer Service Issue",
    1673983: "General Complaint / Other",
    1942580: "Cancellation / Switching / Retention",
    285307: "Network / Connectivity",
    1325693: "Website / App Issue",
    1698067: "Device / Technical Issue",
    1997288: "Order / Upgrade / iPhone",
    241452: "Customer Service Issue",
    373531: "Website / App Issue",
    771500: "Network / Connectivity",
    1393508: "Order / Upgrade / iPhone",
    1968477: "Network / Connectivity",

    # Additional 8 corrections
    740879: "Customer Service Issue",
    1323049: "Order / Upgrade / iPhone",
    2362032: "Device / Technical Issue",
    2368646: "Billing / Payment",
    2802623: "General Complaint / Other",
    1599651: "Device / Technical Issue",
    1536586: "Device / Technical Issue",
    1327222: "Order / Upgrade / iPhone",
}

df["tweet_id_num"] = pd.to_numeric(
    df["customer_tweet_id"],
    errors="coerce"
).astype("Int64")

changes = 0

for tweet_id, new_label in corrections.items():

    mask = df["tweet_id_num"] == tweet_id

    if mask.any():

        old_label = df.loc[mask, "intent"].iloc[0]

        df.loc[mask, "intent"] = new_label

        changes += 1

        print(
            f"{tweet_id}: "
            f"{old_label} -> {new_label}"
        )

    else:
        print(
            f"WARNING: Tweet ID {tweet_id} not found"
        )

df = df.drop(columns=["tweet_id_num"])

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("REVIEW COMPLETE")
print("=" * 60)

print("Rows:", len(df))
print("Corrections applied:", changes)

print(
    "Missing intent:",
    df["intent"].isna().sum()
)

print(
    "Missing escalation:",
    df["escalation"].isna().sum()
)

print("\nNew intent distribution:")
print(df["intent"].value_counts())

print("\nEscalation distribution:")
print(df["escalation"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)