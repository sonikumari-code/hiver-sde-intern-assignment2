import pandas as pd

INPUT = "results/human_review_30.csv"
OUTPUT = "results/human_review_30_scored.csv"

df = pd.read_csv(INPUT)

df["relevance"] = [1,5,2,3,5,4,1,2,3,1,2,3,4,4,5,1,4,5,3,1,3,5,4,4,3,3,5,3,5,3]

df["groundedness"] = [5,5,2,4,5,5,1,3,4,1,2,5,5,5,5,1,5,5,4,1,4,5,5,5,2,4,5,4,5,4]

df["helpfulness"] = [1,4,1,2,4,3,1,1,2,1,1,3,3,3,4,1,4,4,3,1,2,4,3,4,2,3,4,2,4,2]

df["tone"] = [3,4,3,4,4,4,3,4,4,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]

df["overall"] = [1,4,1,2,4,3,1,2,2,1,2,3,3,3,4,1,4,4,3,1,2,4,3,4,2,3,4,2,4,2]

df["human_notes"] = [
"Polite but completely addresses orders instead of the network-speed issue.",
"Directly addresses the website problem and matches historical evidence.",
"Generic reply does not address pricing or multiple-line concerns.",
"Does not address the fee or account concern.",
"Relevant troubleshooting questions for the connectivity issue.",
"Related to switching but does not directly investigate the bill increase.",
"Unrelated to the cancellation or switching complaint.",
"Polite but does not address the chat-loading problem.",
"Offers DM but does not address the serious complaint.",
"Unrelated to the billing and discount problem.",
"Polite but incorrectly assumes a device preorder issue.",
"Grounded in a similar upgrade conversation but vague.",
"Reasonable escalation but gives little network-specific help.",
"Appropriate for a vague complaint but not very specific.",
"Directly related to the checkout problem.",
"Unrelated to the technical iPhone problem.",
"Acknowledges complaint and gives a clear DM next step.",
"Relevant network troubleshooting based on similar evidence.",
"Reasonable generic support response but lacks exact issue.",
"About device preorder instead of switching carriers.",
"Customer gives very little context, so clarification is reasonable.",
"Appropriate response to a customer who may leave.",
"Acknowledges poor service and offers investigation.",
"Reasonable secure-DM next step.",
"Polite but context is unclear.",
"Useful eligibility link but limited context.",
"Directly addresses customer-service complaint.",
"Only a URL is provided, so clarification is reasonable.",
"Addresses customer-service complaint and offers help.",
"Only a URL is provided, so clarification is reasonable."
]

df.to_csv(OUTPUT, index=False)

print("SUCCESS")
print("Rows:", len(df))
print("Scored rows:", df["overall"].notna().sum())
print("Average overall:", round(df["overall"].mean(), 2))
print("Saved:", OUTPUT)
