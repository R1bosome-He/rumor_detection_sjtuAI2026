"""
BiLSTM + Attention 模型 — 推理与评估
────────────────────────────────────
用法:
    # 单条推理
    python -m inference.inference_bilstm

    # 批量评估 (读 val.csv)
    python -m inference.inference_bilstm --eval
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from model.bilstm_attention import BiLSTMAttention
from train.train_bilstm import (
    BILSTM_MODEL_DIR,
    PAD_IDX,
    UNK_IDX,
    BiLSTMDataset,
    _tokenize,
)
from utils.config import VAL_PATH


class BiLSTMClassifier:
    """加载训练好的 BiLSTM + Attention 模型，提供与 RumourDetectClass 一致的接口"""

    def __init__(self, model_dir=None):
        model_dir = Path(model_dir or BILSTM_MODEL_DIR)

        # 加载 checkpoint
        ckpt = torch.load(
            model_dir / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,  # 允许 vocab dict
        )
        self.vocab = ckpt["vocab"]
        self.config = ckpt["config"]

        # 加载阈值
        with open(model_dir / "threshold.json", "r", encoding="utf-8") as f:
            threshold_info = json.load(f)
        self.threshold = threshold_info.get("threshold", 0.5)

        # 重建模型并加载权重
        self.model = BiLSTMAttention(
            vocab_size=len(self.vocab),
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def classify(self, text: str) -> int:
        """二分类: 0=非谣言, 1=谣言"""
        tokens = _tokenize(str(text))
        ids = [
            self.vocab.get(t, UNK_IDX)
            for t in tokens[: self.config["max_len"]]
        ]
        seq_len = len(ids)
        ids += [PAD_IDX] * (self.config["max_len"] - seq_len)
        mask = [True] * seq_len + [False] * (self.config["max_len"] - seq_len)

        input_ids = torch.tensor([ids], dtype=torch.long)
        attn_mask = torch.tensor([mask], dtype=torch.bool)

        with torch.no_grad():
            logits, _ = self.model(input_ids, attention_mask=attn_mask)
            probs = torch.softmax(logits, dim=-1)[0]
            rumor_prob = probs[1].item()

        return int(rumor_prob >= self.threshold)


def evaluate(val_path=None):
    """批量评估"""
    if val_path is None:
        val_path = VAL_PATH
    print(f"评估数据集: {val_path}")

    clf = BiLSTMClassifier()
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

    # 单条推理 demo
    clf = BiLSTMClassifier()
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
