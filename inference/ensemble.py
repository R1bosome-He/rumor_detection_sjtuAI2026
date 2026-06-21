"""
三模型投票集成 — 谣言检测
─────────────────────────
合并 RoBERTa + BiLSTM + TF-IDF 三个模型的预测结果，
采用多数投票 (2/3) 决定最终标签。

用法:
    # 单条推理
    python -m inference.ensemble

    # 批量评估 (val.csv)
    python -m inference.ensemble --eval

    # 评估 JSONL
    python -m inference.ensemble --eval path/to/data.json
"""

import json
import sys
from collections import Counter
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

from utils.config import VAL_PATH


# ═══════════════════════════════════════════════════════════════════════
# 模型加载器 (惰性加载, 失败不阻塞)
# ═══════════════════════════════════════════════════════════════════════

_models = {}  # 缓存单例


def _load_roberta():
    """加载 RoBERTa 模型"""
    from inference.inference import RumourDetectClass
    return RumourDetectClass()


def _load_bilstm():
    """加载 BiLSTM + Attention 模型"""
    from inference.inference_bilstm import BiLSTMClassifier
    return BiLSTMClassifier()


def _load_tfidf():
    """加载 TF-IDF + LR 模型"""
    from inference.inference_tfidf import TFIDFClassifier
    return TFIDFClassifier()


LOADERS = {
    "roberta": _load_roberta,
    "bilstm":  _load_bilstm,
    "tfidf":   _load_tfidf,
}


def _get_model(name: str):
    """惰性加载单个模型, 失败返回 None"""
    if name not in _models:
        try:
            print(f"  加载 {name} ...")
            _models[name] = LOADERS[name]()
        except Exception as e:
            print(f"  [跳过] {name}: {e}")
            _models[name] = None
    return _models[name]


# ═══════════════════════════════════════════════════════════════════════
# 投票逻辑
# ═══════════════════════════════════════════════════════════════════════


# ── 模型权重（加权投票）──────────────────────────────────────────────
# RoBERTa 权重更高 (预训练语义知识丰富), BiLSTM 为辅 (上下文建模)

MODEL_WEIGHTS = {
    "roberta": 2,
    "bilstm":  1,
}


class VotingEnsemble:
    """双模型加权投票 (RoBERTa:2, BiLSTM:1)"""

    def __init__(self, models=None, weights=None):
        """
        Args:
            models:  启用的模型名列表, 默认 ["roberta", "bilstm"]
            weights: 模型权重字典, 默认使用 MODEL_WEIGHTS
        """
        self.model_names = models or ["roberta", "bilstm"]
        self.weights = weights or MODEL_WEIGHTS
        self.active = []

        print("Voting Ensemble 初始化 (加权投票) ...")
        for name in self.model_names:
            m = _get_model(name)
            if m is not None:
                w = self.weights.get(name, 1)
                self.active.append((name, m, w))
                print(f"  {name}: weight={w}")

        if len(self.active) < 2:
            print(f"  警告: 仅 {len(self.active)} 个模型可用, "
                  "投票退化为单模型决策")

        total_w = sum(w for _, _, w in self.active)
        print(f"  总权重: {total_w}, "
              f"活跃: {[(n, w) for n, _, w in self.active]}\n")

    def predict_one(self, text: str) -> dict:
        """单条加权投票

        Returns:
            dict with keys: label, confidence, votes, details
        """
        weighted_votes = {0: 0, 1: 0}
        raw_votes = []
        details = {}

        for name, model, weight in self.active:
            try:
                pred = model.classify(str(text))
                weighted_votes[pred] += weight
                raw_votes.append(pred)
                details[name] = pred
            except Exception as e:
                details[name] = f"error: {e}"

        total_weight = weighted_votes[0] + weighted_votes[1]
        if total_weight == 0:
            return {"label": 0, "confidence": 0.0,
                    "votes": raw_votes, "details": details}

        winner = 1 if weighted_votes[1] >= weighted_votes[0] else 0
        conf = weighted_votes[winner] / total_weight

        return {
            "label": winner,
            "confidence": conf,
            "votes": raw_votes,
            "weighted": dict(weighted_votes),
            "details": details,
        }

    def predict(self, texts) -> list:
        """批量投票"""
        return [self.predict_one(str(t)) for t in texts]


def evaluate(val_path=None):
    """在验证集上评估集成效果"""
    val_path = Path(val_path or VAL_PATH)

    # 支持 CSV 和 JSONL
    if val_path.suffix == ".csv":
        df = pd.read_csv(val_path)
        texts = df["text"].tolist()
        labels = df["label"].astype(int).tolist()
    else:
        texts, labels = [], []
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                if not obj:
                    continue
                texts.append(obj["text"])
                labels.append(int(obj["label"]))

    print(f"评估: {val_path} ({len(texts)} 条)")

    ensemble = VotingEnsemble()
    results = ensemble.predict(texts)
    y_pred = [r["label"] for r in results]

    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average="binary")
    cm = confusion_matrix(labels, y_pred)

    # 统计一致率
    total_weight = sum(ensemble.weights.get(n, 0) for n in ensemble.model_names)
    unanimous = sum(1 for r in results if r["confidence"] == 1.0)
    roberta_wins = sum(1 for r in results
                       if r.get("weighted", {}).get(1, 0) == 2
                       and r.get("weighted", {}).get(0, 0) == 1)
    bilstm_wins = sum(1 for r in results
                      if r.get("weighted", {}).get(1, 0) == 1
                      and r.get("weighted", {}).get(0, 0) == 2)
    print(f"\n投票统计 (RoBERTa:2 BiLSTM:1, 总权重{total_weight}): "
          f"一致 {unanimous}/{len(results)} "
          f"({unanimous/len(results)*100:.1f}%), "
          f"RoBERTa独决 {roberta_wins}/{len(results)}, "
          f"BiLSTM独决 {bilstm_wins}/{len(results)}")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(labels, y_pred, target_names=['non-rumor', 'rumor'])}")

    return {"accuracy": acc, "f1": f1, "results": results}


def main():
    if "--eval" in sys.argv:
        # 支持 python -m inference.ensemble --eval [path]
        idx = sys.argv.index("--eval")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        evaluate(path)
        return

    ensemble = VotingEnsemble()

    test_texts = [
        "BREAKING: #Ferguson police chief just announced that "
        "officer Darren Wilson shot the unarmed teen, Michael Brown.",
        "Swiss museum confirms it will take on #Gurlitt collection",
    ]
    for text in test_texts:
        r = ensemble.predict_one(text)
        label_name = "谣言" if r["label"] else "非谣言"
        print(f"[{r['label']}] {label_name}  conf={r['confidence']:.2f}")
        print(f"  原始票: {r['votes']}  加权: {r.get('weighted', 'N/A')}")
        print(f"  详情: {r['details']}")
        print(f"  {text[:80]}...\n")


if __name__ == "__main__":
    main()
