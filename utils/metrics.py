"""Metrics and threshold search"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }

def find_best_threshold(logits, labels):
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    labels = np.asarray(labels)

    sorted_probs = np.sort(np.unique(probs))
    midpoints = (sorted_probs[:-1] + sorted_probs[1:]) / 2 if len(sorted_probs) > 1 else np.array([])
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], sorted_probs, midpoints)))

    best = None
    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        key = (acc, f1, -abs(float(threshold) - 0.5))
        if best is None or key > best["key"]:
            best = {
                "threshold": float(threshold),
                "accuracy": float(acc),
                "f1": float(f1),
                "preds": preds,
                "key": key,
            }

    return best