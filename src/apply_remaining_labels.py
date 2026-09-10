import pandas as pd

input_file = "data/golden_200_labeled.csv"
output_file = "data/golden_200_labeled.csv"

df = pd.read_csv(input_file)

labels = [
    ("General Complaint / Other", "Human"),          # 150
    ("General Complaint / Other", "Human"),          # 151
    ("Network / Connectivity", "Human"),             # 152
    ("Customer Service Issue", "Human"),             # 153
    ("Device / Technical Issue", "Human"),           # 154
    ("Network / Connectivity", "Human"),             # 155
    ("Network / Connectivity", "Human"),             # 156
    ("Billing / Payment", "Human"),                  # 157
    ("General Complaint / Other", "Human"),          # 158
    ("Network / Connectivity", "Human"),             # 159
    ("General Complaint / Other", "Human"),          # 160
    ("Customer Service Issue", "Human"),             # 161
    ("Order / Upgrade / iPhone", "Human"),           # 162
    ("Order / Upgrade / iPhone", "Human"),           # 163
    ("Billing / Payment", "Human"),                  # 164
    ("Network / Connectivity", "Human"),             # 165
    ("Network / Connectivity", "Human"),             # 166
    ("Order / Upgrade / iPhone", "Human"),           # 167
    ("Billing / Payment", "Human"),                  # 168
    ("General Complaint / Other", "Human"),          # 169
    ("General Complaint / Other", "Human"),          # 170
    ("Device / Technical Issue", "Human"),           # 171
    ("Network / Connectivity", "Human"),             # 172
    ("Network / Connectivity", "Human"),             # 173
    ("General Complaint / Other", "Auto"),           # 174
    ("Customer Service Issue", "Human"),             # 175
    ("Billing / Payment", "Human"),                  # 176
    ("Network / Connectivity", "Human"),             # 177
    ("Network / Connectivity", "Human"),             # 178
    ("Network / Connectivity", "Human"),             # 179
    ("Network / Connectivity", "Human"),             # 180
    ("Customer Service Issue", "Human"),             # 181
    ("General Complaint / Other", "Human"),          # 182
    ("Network / Connectivity", "Human"),             # 183
    ("Customer Service Issue", "Human"),             # 184
    ("General Complaint / Other", "Human"),          # 185
    ("Customer Service Issue", "Human"),             # 186
    ("Order / Upgrade / iPhone", "Human"),           # 187
    ("Device / Technical Issue", "Human"),           # 188
    ("Customer Service Issue", "Human"),             # 189
    ("Order / Upgrade / iPhone", "Human"),           # 190
    ("Customer Service Issue", "Human"),             # 191
    ("Customer Service Issue", "Human"),             # 192
    ("Order / Upgrade / iPhone", "Human"),           # 193
    ("Network / Connectivity", "Human"),             # 194
    ("Network / Connectivity", "Human"),             # 195
    ("Customer Service Issue", "Human"),             # 196
    ("Order / Upgrade / iPhone", "Human"),           # 197
    ("Network / Connectivity", "Human"),             # 198
    ("Billing / Payment", "Human"),                  # 199
    ("Billing / Payment", "Human"),                  # 200
]

# Fill rows 150-200
for i, (intent, escalation) in enumerate(labels, start=149):
    df.loc[i, "intent"] = intent
    df.loc[i, "escalation"] = escalation

df.to_csv(output_file, index=False)

print("=" * 50)
print("Golden Set Completed")
print("=" * 50)
print("Total rows:", len(df))
print("Rows labeled:", df["intent"].notna().sum())
print("Rows remaining:", df["intent"].isna().sum())
print("Output:", output_file)
print("=" * 50)