import pandas as pd

HISTORY_FILE = "data/sprintcare_pairs.csv"
GOLDEN_FILE = "data/golden_200_reviewed.csv"
OUTPUT_FILE = "data/sprintcare_history_clean.csv"

history = pd.read_csv(HISTORY_FILE)
golden = pd.read_csv(GOLDEN_FILE)

golden_ids = set(
    golden["customer_tweet_id"].astype(str)
)

history["customer_tweet_id"] = (
    history["customer_tweet_id"].astype(str)
)

clean_history = history[
    ~history["customer_tweet_id"].isin(golden_ids)
].copy()

clean_history = clean_history.reset_index(drop=True)

print("=" * 60)
print("REMOVING GOLDEN EXAMPLES FROM HISTORY")
print("=" * 60)

print("Original history rows:", len(history))
print("Golden test rows:", len(golden))
print("Clean history rows:", len(clean_history))

remaining_golden = clean_history[
    clean_history["customer_tweet_id"].isin(golden_ids)
]

print("Golden IDs remaining in history:", len(remaining_golden))

clean_history.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 60)