"""Batch evaluation — reusable function, accepts any CSV with 'text' + 'label' columns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from utils.config import VAL_PATH
from inference.inference import RumourDetectClass


def evaluate(val_path=None):
    """Evaluate the rumor detection model on a CSV file.

    Parameters
    ----------
    val_path : str or Path, optional
        Path to a CSV with 'text' and 'label' columns.
        Defaults to the project validation set.

    Returns
    -------
    dict with keys: accuracy, f1, y_true, y_pred, report (str), confusion (array)
    """
    if val_path is None:
        val_path = VAL_PATH

    clf = RumourDetectClass()
    val_df = pd.read_csv(val_path)
    y_true = val_df["label"].to_numpy()
    y_pred = np.array([clf.classify(str(t)) for t in val_df["text"]])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["non-rumor", "rumor"])

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{report}")

    return {
        "accuracy": acc,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report,
        "confusion": cm,
    }


def main():
    evaluate()


if __name__ == "__main__":
    main()
