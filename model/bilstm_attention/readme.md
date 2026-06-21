# BiLSTM + Attention 谣言检测

基于 BiLSTM + 加性注意力的谣言检测模型，数据格式为 JSONL。

## 架构

```
Text → Embedding (200d) → BiLSTM (2层, 128d) → Attention → FC → Rumor/Non-rumor
```

## 优化：GloVe 预训练词嵌入

**问题：** 随机初始化的 `nn.Embedding` 需要从零学习每个词的语义，在小数据集上容易欠拟合。

**方案：** 使用 GloVe Twitter 200d（在 270 亿推文 token 上预训练）初始化 Embedding 层，训练时继续微调（`freeze=False`）。

**原理：** "shot" 和 "fired" 在 GloVe 空间中已接近，模型不需要从零学习它们的语义关联，只需学会它们在谣言语境中的模式。

**实现：** 首次训练时自动下载 `glove.twitter.27B.zip` → 解压 → 逐行匹配词汇表 → `nn.Embedding.from_pretrained()` 注入。

覆盖率：~85%（~3200/3800 词命中）。

## 快速开始

### 训练

```bash
python -m train.train_bilstm
```

模型与词汇表保存到 `model/bilstm_attention/`：

- `checkpoint.pt` — 完整权重快照
- `vocab.json` — 词 → 索引映射
- `threshold.json` — 最优分类阈值与评估指标

### 评估

```bash
python -m inference.inference_bilstm --eval
```

输出 accuracy、F1、混淆矩阵、classification report。

### 单条推理

```bash
python -m inference.inference_bilstm
```

## 数据格式

`dataset/split/train.json` 与 `val.json`，每行一个 JSON：

```json
{"id":"...","text":"推文内容","label":1,"event":0}
```

- `label`: 0 = 非谣言, 1 = 谣言
- `event`: 事件 ID（训练中未使用）

## 配置

在 `train/train_bilstm.py` 顶部 `BILSTM_CONFIG` 字典中修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_dim` | 200 | 词嵌入维度 |
| `hidden_dim` | 128 | LSTM 隐藏层大小 |
| `num_layers` | 2 | BiLSTM 层数 |
| `dropout` | 0.5 | Dropout 比例 |
| `max_len` | 64 | 推文最大词数 |
| `min_freq` | 2 | 词表最低词频 |
| `batch_size` | 64 | 批大小 |
| `learning_rate` | 1e-3 | 初始学习率 |
| `early_stop_patience` | 5 | 早停耐心值 |

## 依赖

```bash
pip install torch numpy scikit-learn pandas
```

## 预测流程

1. JSONL 加载 → 抽取 `text` 字段
2. `_tokenize()`: 小写 + 分词
3. `vocab` 映射: token → 整数索引 (填充/截断到 max_len)
4. `BiLSTMAttention.forward()`: Embedding → BiLSTM → Attention → FC
5. softmax 概率 ≥ threshold → 1 (谣言) / 0 (非谣言)
