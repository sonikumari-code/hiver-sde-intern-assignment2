import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


GOLDEN_FILE = "data/golden_200_reviewed.csv"

MODEL_FILE = "results/clean_intent_model.joblib"
VECTORIZER_FILE = "results/clean_tfidf_vectorizer.joblib"

PREDICTIONS_FILE = "results/clean_golden_predictions.csv"
CONFUSION_FILE = "results/clean_golden_confusion_matrix.csv"


print("=" * 60)
print("CLEAN INTENT MODEL EVALUATION")
print("=" * 60)


# Load golden test set
golden = pd.read_csv(GOLDEN_FILE)

golden = golden.dropna(
    subset=["customer_message", "intent"]
).reset_index(drop=True)


# Load clean model and vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)


# Prepare test data
X_test = golden["customer_message"].astype(str)
y_test = golden["intent"].astype(str)


# Transform test messages
X_test_tfidf = vectorizer.transform(X_test)


# Predict
y_pred = model.predict(X_test_tfidf)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print("\n============================================================")
print("INTENT RESULTS")
print("============================================================")

print("Intent Accuracy:", round(accuracy, 4))
print("Intent Accuracy %:", round(accuracy * 100, 2))
print("Intent Macro F1:", round(macro_f1, 4))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Save predictions
predictions = golden.copy()

predictions["predicted_intent"] = y_pred

predictions["correct"] = (
    predictions["intent"]
    == predictions["predicted_intent"]
)

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# Save confusion matrix
labels = sorted(golden["intent"].unique())

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
    CONFUSION_FILE
)


print("\n============================================================")
print("FILES SAVED")
print("============================================================")

print("Predictions:", PREDICTIONS_FILE)
print("Confusion Matrix:", CONFUSION_FILE)

print("\nClean evaluation completed successfully.")

print("=" * 60)