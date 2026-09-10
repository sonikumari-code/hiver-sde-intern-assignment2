import pandas as pd
import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# Files
# --------------------------------------------------

TRAIN_FILE = "data/sprintcare_training.csv"
GOLDEN_FILE = "data/golden_200_labeled.csv"

MODEL_FILE = "results/intent_model.joblib"
VECTORIZER_FILE = "results/tfidf_vectorizer.joblib"


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

print("=" * 60)
print("Training on Historical SprintCare Data")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
golden_df = pd.read_csv(GOLDEN_FILE)

train_df = train_df.dropna(subset=["customer_message", "intent"])
golden_df = golden_df.dropna(subset=["customer_message", "intent"])

print("Training examples:", len(train_df))
print("Golden test examples:", len(golden_df))


# --------------------------------------------------
# 2. Prepare X and y
# --------------------------------------------------

X_train = train_df["customer_message"].astype(str)
y_train = train_df["intent"].astype(str)

X_test = golden_df["customer_message"].astype(str)
y_test = golden_df["intent"].astype(str)


# --------------------------------------------------
# 3. TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training matrix:", X_train_tfidf.shape)
print("Test matrix:", X_test_tfidf.shape)


# --------------------------------------------------
# 4. Train classifier
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# --------------------------------------------------
# 5. Predict Golden 200
# --------------------------------------------------

print("Evaluating on Golden 200...")

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# 6. Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 7. Confusion Matrix
# --------------------------------------------------

labels = sorted(y_test.unique())

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

cm_df.to_csv(
    "results/golden_confusion_matrix.csv"
)


# --------------------------------------------------
# 8. Save predictions
# --------------------------------------------------

results_df = golden_df[
    [
        "customer_tweet_id",
        "customer_message",
        "intent"
    ]
].copy()

results_df["predicted_intent"] = y_pred
results_df["correct"] = (
    results_df["intent"] ==
    results_df["predicted_intent"]
)

results_df.to_csv(
    "results/golden_predictions.csv",
    index=False
)


# --------------------------------------------------
# 9. Save model
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)


# --------------------------------------------------
# 10. Save metrics
# --------------------------------------------------

metrics = {
    "evaluation_set": "golden_200",
    "training_examples": len(train_df),
    "test_examples": len(golden_df),
    "accuracy": accuracy,
    "classification_report": report
}

with open(
    "results/golden_intent_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)


# --------------------------------------------------
# Done
# --------------------------------------------------

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print("Model       :", MODEL_FILE)
print("Vectorizer  :", VECTORIZER_FILE)
print("Metrics     : results/golden_intent_metrics.json")
print("Predictions : results/golden_predictions.csv")
print("Confusion   : results/golden_confusion_matrix.csv")

print("=" * 60)