"""
TF-IDF + Logistic Regression 模型 — 推理与评估
──────────────────────────────────────────────
用法:
    # 单条推理
    python -m inference.inference_tfidf

    # 批量评估
    python -m inference.inference_tfidf --eval
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from utils.config import BASE_DIR, VAL_PATH
from utils.data import preprocess_tweet

MODEL_DIR = BASE_DIR / "model" / "tfidf_lr"


class TFIDFClassifier:
    """加载训练好的 TF-IDF + LR 模型"""

    def __init__(self, model_dir=None):
        model_dir = Path(model_dir or MODEL_DIR)

        with open(model_dir / "pipeline.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(model_dir / "classifier.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(model_dir / "threshold.json", "r", encoding="utf-8") as f:
            self.threshold = json.load(f)["threshold"]

    def classify(self, text: str) -> int:
        """二分类: 0=非谣言, 1=谣言"""
        clean = preprocess_tweet(str(text))
        vec = self.vectorizer.transform([clean])
        prob = self.model.predict_proba(vec)[0, 1]
        return int(prob >= self.threshold)


def evaluate(val_path=None):
    """批量评估"""
    val_path = Path(val_path or VAL_PATH)
    print(f"评估数据集: {val_path}")

    clf = TFIDFClassifier()
    val_df = pd.read_csv(val_path)
    y_true = val_df["label"].astype(int).tolist()
    y_pred = [clf.classify(str(t)) for t in val_df["text"]]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['non-rumor', 'rumor'])}")

    return {"accuracy": acc, "f1": f1, "y_true": y_true, "y_pred": y_pred}


def main():
    if "--eval" in sys.argv:
        evaluate()
        return

    clf = TFIDFClassifier()
    test_texts = [
        "BREAKING: #Ferguson police chief just announced that officer Darren Wilson shot the unarmed teen, Michael Brown.",
        "Swiss museum confirms it will take on #Gurlitt collection",
    ]
    for text in test_texts:
        label = clf.classify(text)
        label_name = "谣言" if label else "非谣言"
        print(f"[{label}] {label_name}")
        print(f"  {text[:80]}...\n")


if __name__ == "__main__":
    main()
