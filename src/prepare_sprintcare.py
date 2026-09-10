import pandas as pd

INPUT_FILE = "data/twcs.csv"
OUTPUT_FILE = "data/sprintcare_pairs.csv"

# Step 1: Read the complete dataset in chunks
chunks = []

for chunk in pd.read_csv(INPUT_FILE, chunksize=100000):
    # Keep all SprintCare tweets
    sprintcare_rows = chunk[chunk["author_id"] == "sprintcare"]

    if not sprintcare_rows.empty:
        chunks.append(sprintcare_rows)

sprintcare = pd.concat(chunks, ignore_index=True)

print("SprintCare tweets:", len(sprintcare))

# Step 2: Collect IDs of tweets that SprintCare replied to
customer_ids = set(
    sprintcare["in_response_to_tweet_id"]
    .dropna()
    .astype(int)
)

print("Customer tweet IDs found:", len(customer_ids))

# Step 3: Read the original dataset again
pairs = []

for chunk in pd.read_csv(INPUT_FILE, chunksize=100000):

    # Customer tweets whose IDs were referenced by SprintCare
    customers = chunk[
        chunk["tweet_id"].isin(customer_ids)
    ]

    if not customers.empty:
        pairs.append(customers)

customers = pd.concat(pairs, ignore_index=True)

print("Customer tweets found:", len(customers))

# Step 4: Match customer tweet with SprintCare reply
sprintcare["customer_tweet_id"] = (
    sprintcare["in_response_to_tweet_id"]
    .dropna()
    .astype(int)
)

result = sprintcare.merge(
    customers[["tweet_id", "text"]],
    left_on="customer_tweet_id",
    right_on="tweet_id",
    suffixes=("_brand", "_customer")
)

# Rename columns clearly
result = result.rename(
    columns={
        "text_customer": "customer_message",
        "text_brand": "brand_reply"
    }
)

# Keep useful columns
result = result[
    [
        "customer_tweet_id",
        "customer_message",
        "tweet_id_brand",
        "brand_reply",
        "created_at",
    ]
]

# Remove duplicate pairs
result = result.drop_duplicates()

# Save
result.to_csv(OUTPUT_FILE, index=False)

print("Final conversation pairs:", len(result))
print(f"Saved to: {OUTPUT_FILE}")

# Show examples
print("\nSample conversations:\n")
print(result.head(10).to_string(index=False))