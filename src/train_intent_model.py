import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. Load Golden Dataset
# --------------------------------------------------

input_file = "data/golden_200_labeled.csv"

df = pd.read_csv(input_file)

# Remove any missing rows just in case
df = df.dropna(subset=["customer_message", "intent"])

print("=" * 60)
print("Training Intent Classifier")
print("=" * 60)

print("Total examples:", len(df))
print("Number of intents:", df["intent"].nunique())


# --------------------------------------------------
# 2. Input and Target
# --------------------------------------------------

X = df["customer_message"].astype(str)
y = df["intent"].astype(str)


# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Testing examples :", len(X_test))


# --------------------------------------------------
# 4. TF-IDF Vectorizer
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# 5. Logistic Regression Classifier
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# --------------------------------------------------
# 6. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# --------------------------------------------------
# 8. Confusion Matrix
# --------------------------------------------------

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv("results/confusion_matrix.csv")


# --------------------------------------------------
# 9. Save Metrics
# --------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

metrics = {
    "accuracy": accuracy,
    "training_examples": len(X_train),
    "test_examples": len(X_test),
    "number_of_intents": len(labels),
    "classification_report": report
}

with open("results/intent_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# --------------------------------------------------
# 10. Save Model
# --------------------------------------------------

joblib.dump(model, "results/intent_model.joblib")
joblib.dump(vectorizer, "results/tfidf_vectorizer.joblib")


print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("Model      : results/intent_model.joblib")
print("Vectorizer : results/tfidf_vectorizer.joblib")
print("Metrics    : results/intent_metrics.json")
print("Confusion  : results/confusion_matrix.csv")

print("=" * 60)