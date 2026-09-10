import pandas as pd

HISTORY_FILE = "data/sprintcare_pairs.csv"
GOLDEN_FILE = "data/golden_200_reviewed.csv"
OUTPUT_FILE = "data/sprintcare_history_clean.csv"

history = pd.read_csv(HISTORY_FILE)
golden = pd.read_csv(GOLDEN_FILE)

# IDs ko string mein convert karo
history["customer_tweet_id"] = history["customer_tweet_id"].astype(str)
golden["customer_tweet_id"] = golden["customer_tweet_id"].astype(str)

# Golden IDs
golden_ids = set(golden["customer_tweet_id"])

# Golden messages
golden_messages = set(
    golden["customer_message"]
    .astype(str)
    .str.strip()
)

original_count = len(history)

# Golden tweet IDs remove karo
clean = history[
    ~history["customer_tweet_id"].isin(golden_ids)
].copy()

after_id_removal = len(clean)

# Exact same customer messages bhi remove karo
clean = clean[
    ~clean["customer_message"]
    .astype(str)
    .str.strip()
    .isin(golden_messages)
].copy()

final_count = len(clean)

clean.to_csv(OUTPUT_FILE, index=False)

# Verification
remaining_ids = set(clean["customer_tweet_id"]) & golden_ids

remaining_messages = (
    set(clean["customer_message"].astype(str).str.strip())
    & golden_messages
)

print("=" * 60)
print("CLEAN HISTORY CREATION")
print("=" * 60)

print("Original history rows:", original_count)
print("Golden test rows:", len(golden))
print("After golden ID removal:", after_id_removal)
print("Final clean history rows:", final_count)

print("\nGolden IDs remaining in clean history:", len(remaining_ids))
print("Golden messages remaining in clean history:", len(remaining_messages))

print("\nSaved to:", OUTPUT_FILE)

if len(remaining_ids) == 0 and len(remaining_messages) == 0:
    print("\nSUCCESS: No golden leakage found.")
else:
    print("\nWARNING: Leakage still exists!")