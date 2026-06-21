"""
TF-IDF + Logistic Regression 谣言检测 — 训练脚本
───────────────────────────────────────────────
架构: Text → 预处理 → 词TF-IDF + 字TF-IDF → FeatureUnion → LR → Rumor/Non-rumor

优化: 词级 (1-2 gram) + 字符级 (3-5 gram) 双通道 TF-IDF 融合,
      字符通道捕获 #hashtag 内部结构、ALL_CAPS、typo 等推文特有信号。

用法:
    python -m train.train_tfidf
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion

from utils.config import BASE_DIR
from utils.data import preprocess_tweet

# ── 数据路径 ──────────────────────────────────────────────────────────

TRAIN_JSON = BASE_DIR / "dataset" / "split" / "train.json"
VAL_JSON = BASE_DIR / "dataset" / "split" / "val.json"
MODEL_DIR = BASE_DIR / "model" / "tfidf_lr"

# ── 双通道 TF-IDF + LR 配置 ──────────────────────────────────────────

TFIDF_CONFIG = {
    "name": "tfidf_lr",
    # 词级通道: 捕获 "BREAKING", "police chief", "unarmed teen" 等词/短语
    "word_max_features": 4000,
    "word_ngram": (1, 2),
    # 字符级通道: 捕获 #Hashtag内部结构、ALL_CAPS、typo
    "char_max_features": 2000,
    "char_ngram": (3, 5),
    # 公共
    "min_df": 2,
    # LR
    "C": 0.8,
    "max_iter": 1500,
    "seed": 42,
}


def _load_jsonl(path):
    """读取 JSONL，返回 (texts, labels)"""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if not obj:
                continue
            texts.append(obj["text"])
            labels.append(int(obj["label"]))
    print(f"  加载 {len(texts)} 条 ({Path(path).name})")
    return texts, labels


def find_best_threshold(probs, labels):
    """网格搜索最优分类阈值"""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    sorted_p = np.sort(np.unique(probs))
    midpoints = (
        (sorted_p[:-1] + sorted_p[1:]) / 2
        if len(sorted_p) > 1
        else np.array([])
    )
    candidates = np.unique(
        np.concatenate(([0.0, 0.5, 1.0], sorted_p, midpoints))
    )
    best = None
    for t in candidates:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        key = (acc, f1, -abs(float(t) - 0.5))
        if best is None or key > best["key"]:
            best = {
                "threshold": float(t),
                "accuracy": float(acc),
                "f1": float(f1),
                "preds": preds,
                "key": key,
            }
    return best


def main():
    config = TFIDF_CONFIG
    np.random.seed(config["seed"])

    # ── 1. 加载数据 ──────────────────────────────────────────────
    print(f"Train: {TRAIN_JSON}")
    print(f"Val:   {VAL_JSON}")
    train_texts, train_labels = _load_jsonl(TRAIN_JSON)
    val_texts, val_labels = _load_jsonl(VAL_JSON)

    # ── 2. 文本预处理 ──────────────────────────────────────────
    print("预处理文本 ...")
    train_clean = [preprocess_tweet(str(t)) for t in train_texts]
    val_clean = [preprocess_tweet(str(t)) for t in val_texts]

    # ── 3. 双通道 TF-IDF ────────────────────────────────────────
    #  词通道: 1-2 gram 词组
    word_tfidf = TfidfVectorizer(
        max_features=config["word_max_features"],
        ngram_range=config["word_ngram"],
        min_df=config["min_df"],
        sublinear_tf=True,
        max_df=0.9,
        lowercase=True,
        stop_words="english",
    )
    #  字通道: 3-5 gram 字符级, 捕获 #tag内部、CAPS、typo
    char_tfidf = TfidfVectorizer(
        max_features=config["char_max_features"],
        ngram_range=config["char_ngram"],
        analyzer="char_wb",        # 词边界内的字符 n-gram
        min_df=config["min_df"],
        sublinear_tf=True,
        max_df=0.9,
        lowercase=True,
    )
    vectorizer = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    print(f"TF-IDF: word_ngram={config['word_ngram']} "
          f"word_max={config['word_max_features']}, "
          f"char_ngram={config['char_ngram']} "
          f"char_max={config['char_max_features']}")
    X_train = vectorizer.fit_transform(train_clean)
    X_val = vectorizer.transform(val_clean)
    print(f"  融合特征维度: {X_train.shape[1]}")

    # ── 4. 训练 Logistic Regression ─────────────────────────────
    print(f"LR: C={config['C']}, max_iter={config['max_iter']}")
    model = LogisticRegression(
        C=config["C"],
        max_iter=config["max_iter"],
        class_weight="balanced",
        random_state=config["seed"],
    )
    model.fit(X_train, train_labels)

    # ── 5. 评估 ─────────────────────────────────────────────────
    val_probs = model.predict_proba(X_val)[:, 1]
    threshold_info = find_best_threshold(val_probs, val_labels)

    argmax_preds = model.predict(X_val)
    argmax_acc = accuracy_score(val_labels, argmax_preds)
    argmax_f1 = f1_score(val_labels, argmax_preds, average="binary")

    print(f"\nArgmax    : accuracy={argmax_acc:.4f}  f1={argmax_f1:.4f}")
    print(
        f"Threshold : accuracy={threshold_info['accuracy']:.4f}  "
        f"f1={threshold_info['f1']:.4f}  "
        f"threshold={threshold_info['threshold']:.4f}"
    )
    print(f"\n{classification_report(val_labels, threshold_info['preds'], target_names=['non-rumor', 'rumor'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(val_labels, threshold_info['preds'])}")

    # ── 6. 保存模型 ────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "pipeline.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    summary = {
        "name": config["name"],
        "word_max_features": config["word_max_features"],
        "char_max_features": config["char_max_features"],
        "argmax_accuracy": float(argmax_acc),
        "argmax_f1": float(argmax_f1),
        "threshold": threshold_info["threshold"],
        "threshold_accuracy": threshold_info["accuracy"],
        "threshold_f1": threshold_info["f1"],
    }
    with open(MODEL_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存至 {MODEL_DIR}")


if __name__ == "__main__":
    main()
