import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Files
# --------------------------------------------------

DATA_FILE = "data/sprintcare_pairs.csv"
MODEL_FILE = "results/retrieval_vectorizer.joblib"


# --------------------------------------------------
# Load historical conversations
# --------------------------------------------------

print("=" * 60)
print("SprintCare Historical Reply Retrieval Agent")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

df = df.dropna(
    subset=["customer_message", "brand_reply"]
).reset_index(drop=True)

print("Historical conversations:", len(df))


# --------------------------------------------------
# Build TF-IDF index
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=50000
)

customer_vectors = vectorizer.fit_transform(
    df["customer_message"].astype(str)
)

joblib.dump(
    vectorizer,
    MODEL_FILE
)


# --------------------------------------------------
# Retrieval function
# --------------------------------------------------

def retrieve_similar_messages(message, top_k=3):

    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        customer_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "similarity": float(similarities[index]),
            "customer_message": df.iloc[index]["customer_message"],
            "brand_reply": df.iloc[index]["brand_reply"]
        })

    return results


# --------------------------------------------------
# Generate grounded response
# --------------------------------------------------

def generate_reply(message):

    results = retrieve_similar_messages(
        message,
        top_k=3
    )

    best = results[0]

    # Conservative threshold
    if best["similarity"] < 0.15:

        return {
            "reply": (
                "Thanks for reaching out. We'd like to "
                "understand the issue better so we can help. "
                "Please send us a DM with your account details."
            ),
            "similarity": best["similarity"],
            "grounded": False,
            "evidence": results
        }

    return {
        "reply": best["brand_reply"],
        "similarity": best["similarity"],
        "grounded": True,
        "evidence": results
    }


# --------------------------------------------------
# Interactive testing
# --------------------------------------------------

print("\nAgent is ready.")
print("Type a customer message.")
print("Type 'exit' to stop.\n")


while True:

    message = input("Customer: ")

    if message.lower() == "exit":
        break

    result = generate_reply(message)

    print("\n--- Agent Result ---")

    print("Similarity:",
          round(result["similarity"], 3))

    print("Grounded:",
          result["grounded"])

    print("\nDraft Reply:")
    print(result["reply"])

    print("\nHistorical Evidence:")

    for i, item in enumerate(
        result["evidence"],
        start=1
    ):
        print(f"\nExample {i}")
        print("Similarity:",
              round(item["similarity"], 3))
        print("Customer:",
              item["customer_message"])
        print("SprintCare:",
              item["brand_reply"])

    print("\n" + "-" * 60)