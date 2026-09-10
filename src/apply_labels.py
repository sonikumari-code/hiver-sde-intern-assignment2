import pandas as pd

# ============================================================
# Load golden dataset
# ============================================================

input_file = "data/golden_200.csv"
output_file = "data/golden_200_labeled.csv"

df = pd.read_csv(input_file)

# Convert label columns to text-compatible columns
# This fixes the pandas float64 error for empty columns.
df["intent"] = df["intent"].astype("object")
df["escalation"] = df["escalation"].astype("object")


# ============================================================
# Labels for the first 149 rows
# ============================================================

labels = [
    ("General Complaint / Other", "Human"),
    ("Billing / Payment", "Human"),
    ("Website / App Issue", "Auto"),
    ("General Complaint / Other", "Human"),
    ("Customer Service Issue", "Human"),
    ("Billing / Payment", "Human"),
    ("Website / App Issue", "Auto"),
    ("General Complaint / Other", "Auto"),
    ("General Complaint / Other", "Human"),
    ("Customer Service Issue", "Human"),
    ("Network / Connectivity", "Human"),
    ("Order / Upgrade / iPhone", "Auto"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("General Complaint / Other", "Human"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Billing / Payment", "Human"),
    ("General Complaint / Other", "Human"),
    ("Billing / Payment", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Customer Service Issue", "Human"),
    ("Billing / Payment", "Human"),
    ("Customer Service Issue", "Human"),
    ("Billing / Payment", "Human"),
    ("Network / Connectivity", "Human"),
    ("General Complaint / Other", "Human"),
    ("Website / App Issue", "Human"),
    ("Customer Service Issue", "Human"),
    ("Customer Service Issue", "Human"),
    ("Device / Technical Issue", "Human"),
    ("Billing / Payment", "Human"),
    ("General Complaint / Other", "Human"),
    ("Order / Upgrade / iPhone", "Auto"),
    ("Website / App Issue", "Auto"),
    ("General Complaint / Other", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Billing / Payment", "Human"),
    ("Customer Service Issue", "Human"),
    ("Network / Connectivity", "Human"),
    ("Billing / Payment", "Human"),
    ("Billing / Payment", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Customer Service Issue", "Human"),
    ("Customer Service Issue", "Human"),
    ("Billing / Payment", "Human"),
    ("Device / Technical Issue", "Auto"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Customer Service Issue", "Human"),
    ("Network / Connectivity", "Human"),
    ("Customer Service Issue", "Human"),
    ("Network / Connectivity", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Customer Service Issue", "Human"),
    ("Network / Connectivity", "Human"),
    ("Device / Technical Issue", "Human"),
    ("Website / App Issue", "Human"),
    ("General Complaint / Other", "Human"),
    ("Device / Technical Issue", "Human"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("General Complaint / Other", "Human"),
    ("General Complaint / Other", "Auto"),
    ("Network / Connectivity", "Human"),
    ("Website / App Issue", "Auto"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Customer Service Issue", "Human"),
    ("Billing / Payment", "Human"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Network / Connectivity", "Human"),
    ("Device / Technical Issue", "Human"),
    ("General Complaint / Other", "Human"),
    ("Billing / Payment", "Human"),
    ("Network / Connectivity", "Human"),
    ("Website / App Issue", "Auto"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Cancellation / Switching / Retention", "Human"),
    ("General Complaint / Other", "Auto"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Website / App Issue", "Auto"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Order / Upgrade / iPhone", "Auto"),
    ("General Complaint / Other", "Auto"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Network / Connectivity", "Human"),
    ("Device / Technical Issue", "Human"),
    ("Customer Service Issue", "Human"),
    ("Network / Connectivity", "Human"),
    ("Device / Technical Issue", "Human"),
    ("General Complaint / Other", "Auto"),
    ("Customer Service Issue", "Human"),
    ("General Complaint / Other", "Human"),
    ("Network / Connectivity", "Human"),
    ("Website / App Issue", "Human"),
    ("Device / Technical Issue", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Billing / Payment", "Human"),
    ("Network / Connectivity", "Human"),
    ("General Complaint / Other", "Human"),
    ("Customer Service Issue", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Network / Connectivity", "Human"),
    ("Billing / Payment", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Billing / Payment", "Human"),
    ("Billing / Payment", "Human"),
    ("General Complaint / Other", "Human"),
    ("General Complaint / Other", "Human"),
    ("Device / Technical Issue", "Human"),
    ("General Complaint / Other", "Auto"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Customer Service Issue", "Human"),
    ("Billing / Payment", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Website / App Issue", "Human"),
    ("Cancellation / Switching / Retention", "Human"),
    ("Billing / Payment", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Customer Service Issue", "Human"),
    ("General Complaint / Other", "Human"),
    ("Network / Connectivity", "Human"),
    ("Customer Service Issue", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Device / Technical Issue", "Human"),
    ("Customer Service Issue", "Human"),
    ("Customer Service Issue", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Network / Connectivity", "Human"),
    ("Network / Connectivity", "Human"),
    ("Customer Service Issue", "Human"),
    ("Order / Upgrade / iPhone", "Human"),
    ("Network / Connectivity", "Human"),
    ("Billing / Payment", "Human"),
]


# ============================================================
# Safety check
# ============================================================

if len(df) < len(labels):
    raise ValueError(
        f"Dataset has only {len(df)} rows, "
        f"but {len(labels)} labels were provided."
    )


# ============================================================
# Apply labels
# ============================================================

for i, (intent, escalation) in enumerate(labels):
    df.loc[i, "intent"] = intent
    df.loc[i, "escalation"] = escalation


# ============================================================
# Save labeled dataset
# ============================================================

df.to_csv(output_file, index=False)


# ============================================================
# Print result
# ============================================================

print("=" * 50)
print("Golden Set Labeling Completed")
print("=" * 50)

print(f"Total rows      : {len(df)}")
print(f"Rows labeled    : {len(labels)}")
print(f"Rows remaining  : {len(df) - len(labels)}")
print(f"Output file     : {output_file}")

print("=" * 50)