"""
train_models.py
Trains 3 supervised ML models on the complaint dataset:
  1. Category prediction   (TF-IDF + Logistic Regression)
  2. Priority prediction   (TF-IDF + Random Forest)
  3. Department prediction (TF-IDF + Logistic Regression)

Saves trained models + vectorizer + label encoders as .pkl files
inside machine_learning/models/

Run: python machine_learning/train_models.py
"""

import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "complaints_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run: python dataset/generate_dataset.py first."
        )
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna()
    return df


def train_and_save(df, target_col, model, model_filename, vectorizer, encoder_filename):
    X_text = df["complaint_text"]
    y_raw = df[target_col]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X = vectorizer.transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n=== {target_col.upper()} MODEL ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0))

    joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
    joblib.dump(encoder, os.path.join(MODELS_DIR, encoder_filename))

    return acc


def main():
    df = load_data()
    print(f"Loaded {len(df)} complaint records.")

    # Shared TF-IDF vectorizer trained on all complaint text
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    vectorizer.fit(df["complaint_text"])
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    # 1. Category prediction -> Logistic Regression
    train_and_save(
        df, "category",
        LogisticRegression(max_iter=1000),
        "category_model.pkl",
        vectorizer,
        "category_encoder.pkl",
    )

    # 2. Priority prediction -> Random Forest
    train_and_save(
        df, "priority",
        RandomForestClassifier(n_estimators=200, random_state=42),
        "priority_model.pkl",
        vectorizer,
        "priority_encoder.pkl",
    )

    # 3. Department prediction -> Logistic Regression
    train_and_save(
        df, "department",
        LogisticRegression(max_iter=1000),
        "department_model.pkl",
        vectorizer,
        "department_encoder.pkl",
    )

    print("\nAll models trained and saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()