import pandas as pd

INPUT_FILE = "results/human_review_30.csv"
OUTPUT_FILE = "results/human_review_30_scored.csv"

# ============================================================
# LOAD FILE
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("FILLING HUMAN REVIEW - PROVISIONAL SCORES")
print("=" * 70)

print("Rows found:", len(df))


# ============================================================
# PROVISIONAL SCORES
#
# These are AI-assisted suggested scores.
# They should be manually verified before being called
# independent human evaluation.
# ============================================================

scores = {

    # 90767 - Network issue but retrieved/draft reply is about orders
    90767: [1, 5, 1, 3, 1,
            "Reply is polite and historically grounded, but addresses orders instead of the network-speed issue."],

    # 1325693 - Website loading problem, reply addresses website issue
    1325693: [5, 5, 4, 4, 4,
              "Reply directly addresses the website problem and is based on a similar historical case."],

    # 214338 - Plan/rate complaint, generic reply
    214338: [2, 2, 1, 3, 1,
             "Generic reply does not address the customer's pricing and multiple-line concern."],

    # 541873 - Complaint about fee/advice, generic emotional response
    541873: [3, 4, 2, 4, 2,
             "Reply acknowledges frustration but does not address the fee or account concern."],

    # 752761 - Network/data problem, reply asks useful diagnostic questions
    752761: [5, 5, 4, 4, 4,
             "Reply is relevant and provides useful troubleshooting questions for the connectivity issue."],

    # 891795 - Bill increase and switching, reply discusses switching plans
    891795: [4, 5, 3, 4, 3,
             "Reply is related to switching but does not directly investigate the unexpected bill increase."],

    # 2957223 - Leaving plan, reply is about hotspot
    2957223: [1, 1, 1, 3, 1,
              "Reply is unrelated to the customer's cancellation/switching complaint."],

    # 2642864 - Chat not loading, reply says anything else
    2642864: [2, 3, 1, 4, 2,
              "Reply is polite but fails to address the chat-loading problem."],

    # 492758 - collections/class-action complaint, generic DM request
    492758: [3, 4, 2, 4, 2,
             "Reply offers a DM but does not address the serious collections/legal complaint."],

    # 489811 - double billing/device activation, reply about messaging app
    489811: [1, 1, 1, 2, 1,
             "Reply is unrelated to the billing and discount problem."],

    # 1923697 - general complaint, reply assumes preorder issue
    1923697: [2, 2, 1, 4, 2,
              "Polite response but incorrectly assumes the issue is a device preorder problem."],

    # 2587689 - vague customer complaint, reply gives upgrade eligibility
    2587689: [3, 5, 3, 4, 3,
              "Reply is grounded in a similar upgrade conversation but may not address the vague complaint."],

    # 46023 - network complaint, reply says supervisor will call
    46023: [4, 5, 3, 4, 3,
            "Reply indicates escalation to a supervisor, but gives little information about the network issue."],

    # 1989558 - general complaint, generic issue question
    1989558: [4, 5, 3, 4, 3,
              "Reply is appropriate for a vague complaint but provides limited concrete help."],

    # 1358353 - iPhone website checkout crash, reply asks checkout question
    1358353: [5, 5, 4, 4, 4,
              "Reply is directly related to the checkout problem and follows a similar historical interaction."],

    # 1698067 - slow new iPhone, reply celebrates order completion
    1698067: [1, 1, 1, 4, 1,
              "Reply is unrelated to the customer's technical problem with the new iPhone."],

    # 883549 - account/service complaint, reply asks for DM
    883549: [4, 5, 4, 4, 4,
            "Reply acknowledges the complaint and gives a clear next step through DM."],

    # 2420430 - 3G/LTE problem, reply asks device model
    2420430: [5, 5, 4, 4, 4,
              "Reply is relevant troubleshooting for a network problem and is grounded in similar evidence."],

    # 1014247 - vague complaint, reply offers DM
    1014247: [3, 4, 3, 4, 3,
              "Reply is a reasonable generic support response but does not identify the exact problem."],

    # 1944036 - wants to switch carrier, reply asks about preorder
    1944036: [1, 1, 1, 4, 1,
              "Reply is about device preorder rather than the customer's desire to switch carriers."],

    # 1827792 - "No", generic reply
    1827792: [3, 4, 2, 4, 2,
              "The customer message lacks context, so a clarification request is reasonable, but the response is generic."],

    # 1646457 - Bye, retention-style reply
    1646457: [5, 5, 4, 4, 4,
              "Reply appropriately recognizes a possible customer departure and offers assistance."],

    # 456194 - service horrible, reply offers DM
    456194: [4, 5, 3, 4, 3,
             "Reply acknowledges poor service and offers a path to investigate, but is not specific."],

    # 903109 - "Sure", reply asks for DM/security
    903109: [4, 5, 4, 4, 4,
             "Reply gives a reasonable secure-DM next step, although the short customer message has limited context."],

    # 1968035 - "Done", response is generic concern message
    1968035: [3, 2, 2, 4, 2,
              "Reply is polite but does not clearly connect to the cancellation context."],

    # 1268159 - "Yes and yes", reply gives eligibility link
    1268159: [3, 4, 3, 4, 3,
              "Reply provides a useful eligibility link, although the short message has limited context."],

    # 256957 - customer service sucks, reply asks why and offers DM
    256957: [5, 5, 4, 4, 4,
             "Reply directly acknowledges the customer-service complaint and offers assistance."],

    # 334279 - only URL, generic response
    334279: [3, 4, 2, 4, 2,
             "Customer message contains only a link, so clarification is reasonable, but the response is very generic."],

    # 241452 - customer service sucks, reply asks for DM
    241452: [5, 5, 4, 4, 4,
             "Reply addresses the customer-service complaint and offers a clear next step."],

    # 42854 - only URL, generic response
    42854: [3, 4, 2, 4, 2,
            "The message contains only a URL, so the generic request for clarification is reasonable."]
}


# ============================================================
# APPLY SCORES
# ============================================================

for index, row in df.iterrows():

    try:
        tweet_id = int(float(row["customer_tweet_id"]))
    except:
        continue

    if tweet_id in scores:

        relevance = scores[tweet_id][0]
        groundedness = scores[tweet_id][1]
        helpfulness = scores[tweet_id][2]
        tone = scores[tweet_id][3]
        overall = scores[tweet_id][4]
        notes = scores[tweet_id][5]

        df.loc[index, "relevance"] = relevance
        df.loc[index, "groundedness"] = groundedness
        df.loc[index, "helpfulness"] = helpfulness
        df.loc[index, "tone"] = tone
        df.loc[index, "overall"] = overall
        df.loc[index, "human_notes"] = notes


# ============================================================
# CONVERT SCORE COLUMNS TO NUMERIC
# ============================================================

score_columns = [
    "relevance",
    "groundedness",
    "helpfulness",
    "tone",
    "overall"
]

for column in score_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# CHECK
# ============================================================

print("\nScore summary:")

for column in score_columns:
    print(
        f"{column}:",
        round(df[column].mean(), 2)
    )


print("\nMissing scores:")

for column in score_columns:
    print(
        column + ":",
        int(df[column].isna().sum())
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("Output:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("These are provisional AI-assisted review scores.")
print("Verify them manually before reporting them as human evaluation.")

print("=" * 70)