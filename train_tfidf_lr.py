"""
TF-IDF + Logistic Regression baseline for rumor detection.

Pipeline:
text
 -> preprocessing
 -> TF-IDF vectorization
 -> Logistic Regression
 -> rumor / non-rumor prediction

Input data format:
Each line is a JSON object:
{"id":"536824152345678912","text":"...","label":1,"event":0}
"""

import json
import re
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


TRAIN_PATH = "dataset/split/train.json"
VAL_PATH = "dataset/split/val.json"


def read_json_lines(path: str) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing.

    For Twitter-like data:
    - replace @username with @user
    - replace URL with http
    - normalize whitespace
    - lowercase text
    """
    text = "" if pd.isna(text) else str(text)

    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+", "http", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def load_dataset():
    train_df = read_json_lines(TRAIN_PATH)
    val_df = read_json_lines(VAL_PATH)

    required_cols = ["id", "text", "label", "event"]
    for name, df in [("train", train_df), ("val", val_df)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")

    train_df["text"] = train_df["text"].apply(preprocess_text)
    val_df["text"] = val_df["text"].apply(preprocess_text)

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    return train_df, val_df


def build_model() -> Pipeline:
    """
    TF-IDF + Logistic Regression model.

    TfidfVectorizer converts text into sparse numerical features.
    LogisticRegression performs binary classification.
    """
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def evaluate(model: Pipeline, x_val, y_val):
    y_pred = model.predict(x_val)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    print("\n===== Evaluation Result =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n===== Classification Report =====")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=["non-rumor", "rumor"],
            zero_division=0,
        )
    )

    print("\n===== Confusion Matrix =====")
    print(confusion_matrix(y_val, y_pred))


def main():
    print("Loading dataset...")
    train_df, val_df = load_dataset()

    x_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()

    x_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()

    print(f"Train size: {len(train_df)}")
    print(f"Val size  : {len(val_df)}")

    print("\nTraining TF-IDF + Logistic Regression model...")
    model = build_model()
    model.fit(x_train, y_train)

    evaluate(model, x_val, y_val)


if __name__ == "__main__":
    main()