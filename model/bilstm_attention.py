"""
BiLSTM + Attention 谣言检测模型
───────────────────────────────
架构: Embedding → BiLSTM → Attention → FC → Rumor/Non-rumor

输入格式 (JSON):
    {"id": "...", "text": "推文内容", "label": 1, "event": 0}
    label: 0 = 非谣言, 1 = 谣言
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """加性注意力 (Bahdanau-style additive attention)

    对 BiLSTM 输出的每个时间步计算注意力权重，
    加权求和得到上下文向量，用于最终分类。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)  BiLSTM 所有时间步输出
            mask:        (batch, seq_len)  bool, True=padding 位置需屏蔽

        Returns:
            context:  (batch, hidden_dim)  注意力加权上下文向量
            weights:  (batch, seq_len)     注意力权重 (供可解释性分析)
        """
        # u: (batch, seq_len, hidden_dim)
        u = torch.tanh(self.W(lstm_output))
        # scores: (batch, seq_len, 1)
        scores = self.v(u)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(scores.squeeze(-1), dim=-1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights


class BiLSTMAttention(nn.Module):
    """BiLSTM + Attention 谣言分类模型

    Text → Embedding → BiLSTM → Attention → FC → 2-class softmax
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 2,
        pad_idx: int = 0,
        pretrained_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        if pretrained_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weight, freeze=False, padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, embed_dim, padding_idx=pad_idx
            )
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2  # 双向拼接
        self.embed_dropout = nn.Dropout(dropout)  # Embedding 层后 dropout, 防过拟合
        self.attention = Attention(lstm_out_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids:      (batch, seq_len)  词索引
            attention_mask: (batch, seq_len)  bool, True=padding

        Returns:
            logits:  (batch, num_classes)
            attn_w:  (batch, seq_len)  注意力权重
        """
        emb = self.embedding(input_ids)           # (batch, seq_len, embed_dim)
        emb = self.embed_dropout(emb)              # Embedding dropout (仅训练时生效)

        lstm_out, _ = self.bilstm(emb)  # (batch, seq_len, 2*hidden_dim)

        # 构造 padding mask: True 表示需屏蔽的位置
        if attention_mask is not None:
            pad_mask = ~attention_mask  # attention_mask: True=有效, pad_mask: True=padding
        else:
            pad_mask = None

        context, attn_w = self.attention(lstm_out, mask=pad_mask)
        logits = self.fc(context)
        return logits, attn_w
