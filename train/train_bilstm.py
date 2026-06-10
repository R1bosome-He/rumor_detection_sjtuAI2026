"""
BiLSTM + Attention 谣言检测 — 训练脚本
─────────────────────────────────────
数据格式 (CSV 或 JSON, 每行):
    {"id": "...", "text": "推文内容", "label": 1, "event": 0}

用法:
    python -m train.train_bilstm
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

from model.bilstm_attention import BiLSTMAttention
from utils.config import BASE_DIR

# BiLSTM 专用数据路径 (JSONL 格式)
TRAIN_JSON = BASE_DIR / "dataset" / "split" / "train.json"
VAL_JSON = BASE_DIR / "dataset" / "split" / "val.json"
from utils.data import preprocess_tweet

# ── BiLSTM 专用配置 ──────────────────────────────────────────────────

BILSTM_CONFIG = {
    "name": "bilstm_attn",
    "embed_dim": 200,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.55,             # 折中: 旧0.5太弱(过拟合), 0.6太强(欠拟合)
    "max_len": 64,
    "min_freq": 2,
    "batch_size": 64,
    "learning_rate": 7e-4,       # 折中: 1e-3太快(震荡), 5e-4太慢
    "num_epochs": 30,
    "weight_decay": 2e-4,        # 折中: 1e-5形同虚设, 1e-3约束过度
    "early_stop_patience": 4,    # 折中: 留出足够时间让模型收敛
    "seed": 42,
}

BILSTM_MODEL_DIR = BASE_DIR / "model" / "bilstm_attention"
GLOVE_CACHE = BILSTM_MODEL_DIR / "glove.twitter.27B.200d.txt"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"

# ── 特殊 token ───────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


# ═══════════════════════════════════════════════════════════════════════
# 词汇表构建
# ═══════════════════════════════════════════════════════════════════════


def build_vocab(texts, min_freq=2):
    """从文本列表中构建词 → 索引映射"""
    counter = Counter()
    for text in texts:
        tokens = _tokenize(text)
        counter.update(tokens)

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    idx = len(vocab)
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if word not in vocab:
            vocab[word] = idx
            idx += 1

    print(f"词汇表大小: {len(vocab)} (min_freq={min_freq})")
    return vocab


def _tokenize(text: str):
    """简易分词: 小写 + 按非字母数字字符切分，保留 @user 和 http 占位符"""
    text = preprocess_tweet(str(text))
    # 按空白 + 标点切分，保留 @user 和 http
    tokens = []
    for token in text.lower().split():
        token = token.strip(".,!?;:\"'()[]{}|/\\<>")
        if token:
            tokens.append(token)
    return tokens


# ═══════════════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════════════


class BiLSTMDataset(Dataset):
    """BiLSTM 数据集: 文本 → 词索引序列 + 标签"""

    def __init__(self, texts, labels, vocab, max_len=64):
        self.vocab = vocab
        self.max_len = max_len
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.input_ids = []
        self.attention_masks = []

        for text in texts:
            ids = [vocab.get(t, UNK_IDX) for t in _tokenize(str(text))]
            # 截断
            ids = ids[:max_len]
            seq_len = len(ids)
            # 填充
            pad_len = max_len - seq_len
            ids += [PAD_IDX] * pad_len
            mask = [True] * seq_len + [False] * pad_len  # True=有效

            self.input_ids.append(ids)
            self.attention_masks.append(mask)

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(self.attention_masks, dtype=torch.bool)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


# ═══════════════════════════════════════════════════════════════════════
# 训练与评估
# ═══════════════════════════════════════════════════════════════════════


def compute_class_weights(labels):
    """类别反比加权，用于平衡损失函数"""
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask=mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids, attention_mask=mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return total_loss / total_samples, total_correct / total_samples, all_logits, all_labels


def find_best_threshold(probs, labels):
    """网格搜索最优分类阈值 (最大化 composite key: acc + f1 + 接近 0.5)"""
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


def _download_glove():
    """下载 GloVe Twitter 200d 并解压，返回 txt 路径"""
    import zipfile
    from urllib.request import urlretrieve

    if GLOVE_CACHE.exists():
        print(f"GloVe 已缓存: {GLOVE_CACHE}")
        return GLOVE_CACHE

    BILSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BILSTM_MODEL_DIR / "glove.zip"
    print(f"下载 GloVe Twitter 200d ({GLOVE_URL}) ...")
    urlretrieve(GLOVE_URL, zip_path)
    print("解压中 ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("glove.twitter.27B.200d.txt", BILSTM_MODEL_DIR)
    zip_path.unlink()  # 删除 zip
    print(f"GloVe 就绪: {GLOVE_CACHE}")
    return GLOVE_CACHE


def load_glove_weights(vocab, embed_dim=200):
    """加载 GloVe 预训练向量，返回 (vocab_size, embed_dim) 权重张量"""
    path = _download_glove()
    print(f"加载 GloVe 向量 -> 词汇表 ({len(vocab)} words) ...")

    weight = torch.randn(len(vocab), embed_dim) * 0.01  # 未命中词用随机初始化
    weight[0] = 0.0  # PAD

    hit = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                weight[vocab[word]] = vec
                hit += 1

    coverage = hit / len(vocab) * 100
    print(f"  GloVe 命中: {hit}/{len(vocab)} ({coverage:.1f}%)")
    return weight


def _load_jsonl(path):
    """读取 JSONL 文件，返回 (texts, labels) 两个列表"""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(int(obj["label"]))
    print(f"  加载 {len(texts)} 条 ({path.name})")
    return texts, labels


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════


def main():
    config = BILSTM_CONFIG
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. 加载数据 (JSONL: 每行一个 JSON 对象) ────────────────
    print(f"Train: {TRAIN_JSON}")
    print(f"Val:   {VAL_JSON}")
    train_texts, train_labels = _load_jsonl(TRAIN_JSON)
    val_texts, val_labels = _load_jsonl(VAL_JSON)

    # ── 2. 构建词汇表 ──────────────────────────────────────────
    vocab = build_vocab(train_texts, min_freq=config["min_freq"])

    # ── 3. 构建 Dataset & DataLoader ────────────────────────────
    train_ds = BiLSTMDataset(train_texts, train_labels, vocab, config["max_len"])
    val_ds = BiLSTMDataset(val_texts, val_labels, vocab, config["max_len"])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        drop_last=False,
    )

    # ── 4. 加载 GloVe 预训练词嵌入 (首次自动下载) ──────────────
    glove_weight = load_glove_weights(vocab, embed_dim=config["embed_dim"])

    # ── 5. 初始化模型 ──────────────────────────────────────────
    model = BiLSTMAttention(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pretrained_weight=glove_weight,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"参数量: {total_params:,} (trainable: {trainable_params:,})")

    # ── 6. 损失函数 & 优化器 ────────────────────────────────────
    class_weights = compute_class_weights(np.array(train_labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # ── 7. 训练循环 ────────────────────────────────────────────
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    print(f"\n{'='*50}")
    print(f"开始训练: {config['name']}")
    print(f"{'='*50}\n")

    for epoch in range(1, config["num_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_logits, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "vocab": vocab,
                "config": config,
                "epoch": epoch,
                "val_logits": val_logits,
                "val_labels": val_labels,
            }
            patience_counter = 0
            status = " ⭐ best"
        else:
            patience_counter += 1
            if patience_counter >= config["early_stop_patience"]:
                print(
                    f"\n早停: {config['early_stop_patience']} epochs 无提升\n"
                )
                break

        print(
            f"Epoch {epoch:2d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            f"{status}"
        )

    # ── 8. 阈值优化 ────────────────────────────────────────────
    probs = torch.softmax(best_state["val_logits"], dim=-1)[:, 1].numpy()
    labels_np = best_state["val_labels"].numpy()
    threshold_info = find_best_threshold(probs, labels_np)

    argmax_preds = best_state["val_logits"].argmax(dim=-1).numpy()
    argmax_acc = accuracy_score(labels_np, argmax_preds)
    argmax_f1 = f1_score(labels_np, argmax_preds, average="binary")

    print(f"\nArgmax    : accuracy={argmax_acc:.4f}  f1={argmax_f1:.4f}")
    print(
        f"Threshold : accuracy={threshold_info['accuracy']:.4f}  "
        f"f1={threshold_info['f1']:.4f}  "
        f"threshold={threshold_info['threshold']:.4f}"
    )

    print(f"\n{classification_report(labels_np, threshold_info['preds'], target_names=['non-rumor', 'rumor'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(labels_np, threshold_info['preds'])}")

    # ── 9. 保存模型 ────────────────────────────────────────────
    BILSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 保存 PyTorch checkpoint
    torch.save(
        {
            "model_state_dict": best_state["model"],
            "vocab": best_state["vocab"],
            "config": best_state["config"],
        },
        BILSTM_MODEL_DIR / "checkpoint.pt",
    )

    # 保存词汇表 (JSON, 便于推理时独立加载)
    with open(BILSTM_MODEL_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(best_state["vocab"], f, ensure_ascii=False)

    # 保存阈值和评估结果
    summary = {
        "name": best_state["config"]["name"],
        "best_epoch": best_state["epoch"],
        "argmax_accuracy": float(argmax_acc),
        "argmax_f1": float(argmax_f1),
        "threshold": threshold_info["threshold"],
        "threshold_accuracy": threshold_info["accuracy"],
        "threshold_f1": threshold_info["f1"],
    }
    with open(BILSTM_MODEL_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存至 {BILSTM_MODEL_DIR}")


if __name__ == "__main__":
    main()
