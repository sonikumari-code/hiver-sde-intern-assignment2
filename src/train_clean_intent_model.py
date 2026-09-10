import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INPUT_FILE = "data/sprintcare_training_clean.csv"

MODEL_FILE = "results/clean_intent_model.joblib"
VECTORIZER_FILE = "results/clean_tfidf_vectorizer.joblib"


print("=" * 60)
print("TRAINING CLEAN INTENT MODEL")
print("=" * 60)


# Load clean weakly-labelled training data
df = pd.read_csv(INPUT_FILE)

df = df.dropna(
    subset=["customer_message", "intent"]
).reset_index(drop=True)

print("Training rows:", len(df))
print("Number of intents:", df["intent"].nunique())


# Input and target
X = df["customer_message"].astype(str)
y = df["intent"].astype(str)


# TF-IDF features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF features:", X_tfidf.shape[1])


# Logistic Regression classifier
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_tfidf, y)


# Save model and vectorizer
joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)


print("\nModel saved:")
print(MODEL_FILE)

print("\nVectorizer saved:")
print(VECTORIZER_FILE)

print("\nTraining completed successfully.")
print("=" * 60)