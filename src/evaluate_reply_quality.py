import pandas as pd
import re

INPUT_FILE = "results/clean_support_agent_evaluation.csv"
OUTPUT_FILE = "results/reply_quality_review.csv"

df = pd.read_csv(INPUT_FILE)

def quality_flags(row):
    customer = str(row["customer_message"])
    reply = str(row["draft_reply"])

    # Basic text checks
    reply_words = len(reply.split())

    generic_phrases = [
        "thanks for reaching out",
        "send us a dm",
        "send a dm",
        "we'd like to understand",
        "how can we help",
        "please send us a dm",
        "we can assist"
    ]

    is_generic = any(
        phrase in reply.lower()
        for phrase in generic_phrases
    )

    # Check whether reply contains some words from customer message
    customer_words = set(
        re.findall(r"[a-zA-Z]{4,}", customer.lower())
    )

    reply_words_set = set(
        re.findall(r"[a-zA-Z]{4,}", reply.lower())
    )

    overlap = customer_words.intersection(reply_words_set)

    lexical_overlap = (
        len(overlap) / max(len(customer_words), 1)
    )

    return pd.Series({
        "reply_word_count": reply_words,
        "generic_reply": is_generic,
        "customer_reply_word_overlap": round(
            lexical_overlap, 3
        )
    })


flags = df.apply(quality_flags, axis=1)

result = pd.concat([df, flags], axis=1)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 60)
print("REPLY QUALITY REVIEW DATASET")
print("=" * 60)

print("Examples:", len(result))

print(
    "Generic replies:",
    int(result["generic_reply"].sum())
)

print(
    "Generic reply %:",
    round(
        result["generic_reply"].mean() * 100,
        2
    )
)

print(
    "Average reply words:",
    round(
        result["reply_word_count"].mean(),
        2
    )
)

print(
    "Average customer/reply lexical overlap:",
    round(
        result["customer_reply_word_overlap"].mean(),
        3
    )
)

print("\nFile saved:")
print(OUTPUT_FILE)

print("=" * 60)