# Rumor-Detection

> 基于 Twitter-RoBERTa 的推文谣言检测，支持注意力证据词提取与 LLM 自然语言解释。

## 📝 项目简介

本项目对 [cardiffnlp/twitter-roberta-base](https://huggingface.co/cardiffnlp/twitter-roberta-base) 进行微调，实现推文（Twitter/X）的谣言二分类。核心特色在于：

- **可解释性**：利用模型最后一层注意力权重提取关键证据词，并结合 LLM 生成自然语言分析，解释"为什么这条推文被判为谣言/非谣言"。
- **开箱即用**：模型权重托管在 HuggingFace，本地缺失时自动下载，无需手动配置。
- **模块化设计**：训练、推理、评估、LLM 解释各自独立，可按需使用。

适用于社交媒体内容审核、舆情分析、虚假信息检测等场景。

## ✨ 核心功能

- [x] **谣言检测**：Twitter-RoBERTa 微调，输出 0（非谣言）/ 1（谣言）+ 置信度
- [x] **证据词提取**：基于 attention 权重自动提取推文中的 5 个关键证据词
- [x] **LLM 解释**：调用 DeepSeek-Reasoner 生成 2-4 句中文分析解释
- [x] **批量评估**：读取 CSV 自动计算 Accuracy、F1、混淆矩阵等指标
- [x] **完整训练**：支持类别加权、阈值搜索、Early Stopping

## 🚀 快速开始

### 环境要求

- Python 3.10+

### 安装

```bash
git clone https://github.com/R1bosome-He/rumor_detection_sjtuAI2026.git
cd rumor_detection_sjtuAI2026

# （可选）创建虚拟环境
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 推理

```bash
# 单条交互式推理（含 LLM 解释）
python main.py

# 批量推理（直接输出标签 + 证据词）
python -m inference.inference
```

模型权重会自动从 [FENGYU21/rumor-detection](https://huggingface.co/FENGYU21/rumor-detection) 下载，无需手动配置。

### 评估

```bash
python inference/eval.py
```

### 训练

```bash
python train/train.py
```

## 📊 性能评估

验证集（401 条推文）上的指标：

| 指标 | 数值 |
|---|---|
| Accuracy | 0.8928 |
| F1-Score | 0.8746 |
| Precision (non-rumor) | 0.89 |
| Recall (non-rumor) | 0.92 |
| Precision (rumor) | 0.89 |
| Recall (rumor) | 0.86 |

最佳分类阈值：0.4981

## 📁 项目结构

```
├── main.py               # 入口：交互式推理 + LLM 解释
├── inference/
│   ├── inference.py       # 模型加载、推理、证据词提取
│   └── eval.py            # 批量评估
├── train/
│   ├── train.py           # 训练流水线
│   └── trainer.py         # WeightedTrainer（类别加权）
├── llm/
│   └── llm.py             # LLM 解释器（DeepSeek-Reasoner）
├── utils/
│   ├── config.py          # 全局配置
│   ├── data.py            # 数据加载与预处理
│   └── metrics.py         # 评估指标与阈值搜索
├── model/
│   └── twitter-roberta/   # 本地模型（训练或下载后生成）
├── dataset/
│   └── split/             # 训练/验证集 CSV
└── requirements.txt
```

