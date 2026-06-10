# Rumor-Detection

> 基于 Twitter-RoBERTa 的推文谣言检测，支持证据词提取、LLM 解释与双模型加权投票。

## 📝 项目简介

本项目实现推文（Twitter/X）的谣言二分类，核心模型为 [cardiffnlp/twitter-roberta-base](https://huggingface.co/cardiffnlp/twitter-roberta-base) 微调。额外实现了 BiLSTM + Attention 与 TF-IDF + LR 两个轻量基线，并通过 RoBERTa : BiLSTM = 2 : 1 加权投票集成，验证集准确率达 **89.3%**。特色在于：

- **可解释性**：利用 attention 权重提取证据词，调用 DeepSeek-Reasoner 生成中文分析
- **开箱即用**：RoBERTa 权重托管在 [FENGYU21/rumor-detection](https://huggingface.co/FENGYU21/rumor-detection)，BiLSTM/TF-IDF 权重随仓库发布
- **模块化设计**：训练、推理、评估、LLM 解释、投票集成各自独立

## 🚀 快速开始

```bash
# 安装
git clone https://github.com/R1bosome-He/rumor_detection_sjtuAI2026.git
cd rumor_detection_sjtuAI2026
pip install -r requirements.txt

# 推理（含 LLM 解释）          # 评估（双模型投票）
python main.py                   python -m inference.ensemble --eval
```

> RoBERTa 权重首次自动从 HuggingFace 下载（~500MB），BiLSTM / TF-IDF 权重已在仓库中。

## 📊 性能

验证集 401 条推文，投票集成（RoBERTa : BiLSTM = 2 : 1）：

| 指标 | 数值 |
|------|------|
| Accuracy | 0.8928 |
| F1-Score | 0.8746 |
| Precision (non-rumor) | 0.89 |
| Recall (non-rumor) | 0.92 |
| Precision (rumor) | 0.89 |
| Recall (rumor) | 0.86 |

三条技术路线对比：

| | RoBERTa | BiLSTM | TF-IDF | **投票集成** |
|---|---|---|---|---|
| 准确率 | — | 86.5% | 84.0% | **89.3%** |
| 参数量 | 125M | 2M | 5k | — |
| 训练时间 | ~15 min (GPU) | ~5 min (CPU) | **~3 sec (CPU)** | — |

## 🗳️ 投票策略设计

我们最初尝试了 RoBERTa + BiLSTM + TF-IDF 三模型等权投票，但准确率反而下降了约 1%。原因在于：

- **TF-IDF** 仅基于词频共现统计，完全不具备语义理解能力，在小样本下其投票近似随机噪声
- **BiLSTM** 虽有序列建模能力，但 200 万参数从头学习，2800 条样本不足以使其学到稳定的谣言语义表征

因此我们去掉 TF-IDF，采用 **RoBERTa : BiLSTM = 2 : 1 加权投票**：RoBERTa 拥有 1.25 亿参数与 5800 万条推文的预训练先验，对推文的语义理解远强于 BiLSTM，赋予更高权重合理。当两模型意见一致时（占 89.3%），直接输出结果；不一致时 RoBERTa 以 2:1 权重压倒 BiLSTM。最终准确率回升至与原 RoBERTa 持平，同时保留了 BiLSTM 作为互补视角的诊断价值。

## 💬 LLM 解释

分类模型输出标签后，系统调用 DeepSeek-Reasoner 对结果进行自然语言解释。流程如下：

```
推文文本 + 分类标签 + 模型提取的 attention 证据词
        │
        ▼
   构造 prompt（指导 LLM 以分析口吻输出 2-4 句中文）
        │
        ▼
   POST → https://models.sjtu.edu.cn/api/v1/chat/completions
        │
        ▼
   返回解释: "该推文使用了带有情绪煽动性的大写词汇 BREAKING，
            且未引用任何官方来源，符合未经验证的突发消息特征。"
```

证据词（如 `BREAKING`、`shot`、`unarmed`）来自 RoBERTa 最后一层 attention 对 `[CLS]` token 的权重，经 BPE 字符合并和停用词过滤后取 top-5，注入 prompt 帮助 LLM 聚焦关键信息。API key 通过环境变量 `LLM_API_KEY` 或在 `main.py` 运行时直接输入设置。

## 📁 项目结构

```
├── main.py                     # RoBERTa 交互入口（含 LLM 解释）
├── inference/
│   ├── inference.py            # RoBERTa 推理 + 证据词提取
│   ├── eval.py                 # RoBERTa 评估
│   ├── inference_bilstm.py     # BiLSTM 推理/评估
│   ├── inference_tfidf.py      # TF-IDF 推理/评估
│   └── ensemble.py             # 加权投票集成
├── train/
│   ├── train.py                # RoBERTa 训练
│   ├── trainer.py              # WeightedTrainer（类别加权）
│   ├── train_bilstm.py         # BiLSTM 训练（含 GloVe 自动下载）
│   └── train_tfidf.py          # TF-IDF 训练（词+字双通道）
├── model/
│   ├── twitter-roberta/        # RoBERTa 权重（本地或 HF 下载）
│   ├── bilstm_attention/       # BiLSTM 权重 + 词汇表 + 阈值
│   │   └── readme.md
│   └── tfidf_lr/               # TF-IDF 向量器 + LR 分类器
│       └── readme.md
├── llm/
│   └── llm.py                  # LLM 解释器（DeepSeek-Reasoner）
├── utils/
│   ├── config.py               # 全局配置
│   ├── data.py                 # 数据加载与预处理
│   └── metrics.py              # 评估指标与阈值搜索
├── dataset/split/              # 训练/验证集（JSONL）
└── requirements.txt
```

## 🔧 全部命令

| 目的 | 命令 |
|------|------|
| 交互推理 (含 LLM) | `python main.py` |
| **投票评估** | `python -m inference.ensemble --eval` |
| RoBERTa 评估 | `python inference/eval.py` |
| BiLSTM 评估 | `python -m inference.inference_bilstm --eval` |
| TF-IDF 评估 | `python -m inference.inference_tfidf --eval` |
| RoBERTa 训练 | `python train/train.py` |
| BiLSTM 训练 | `python -m train.train_bilstm` |
| TF-IDF 训练 | `python -m train.train_tfidf` |

## ⚙️ 配置

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `LLM_API_KEY` | LLM API 密钥 | 运行时空缺则提示输入 |
| `LLM_BASE_URL` | API 地址 | `https://models.sjtu.edu.cn/api/v1` |
| `LLM_MODEL` | 模型名 | `deepseek-reasoner` |

训练超参分别在 `train/train.py`（RoBERTa）、`train/train_bilstm.py`（BiLSTM）、`train/train_tfidf.py`（TF-IDF）的配置字典中修改。
