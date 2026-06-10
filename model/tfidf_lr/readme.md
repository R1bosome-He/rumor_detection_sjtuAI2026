# TF-IDF + Logistic Regression 谣言检测

基于 TF-IDF 词频特征 + 逻辑回归的谣言检测模型。

## 原理

**TF-IDF（词频-逆文档频率）** 将每条推文转换为一个稀疏向量：某个词在当前推文中出现越频繁（TF 高）、在整个语料中出现越少（IDF 高），其权重越大。例如 "BREAKING" 在谣言推文中常见但在非谣言中罕见，TF-IDF 会赋予它高分。

**Logistic Regression** 学习每个 TF-IDF 特征的权重，判断推文属于"谣言"类别的概率。带 L2 正则化防止过拟合，`class_weight="balanced"` 自动处理类别不平衡。

```
Text → preprocess_tweet() → TfidfVectorizer → LogisticRegression → Rumor/Non-rumor
```

## 快速开始

### 训练

```bash
python -m train.train_tfidf
```

模型保存到 `model/tfidf_lr/`：

- `vectorizer.pkl` — TF-IDF 向量化器
- `classifier.pkl` — 逻辑回归分类器
- `threshold.json` — 最优分类阈值

### 评估

```bash
python -m inference.inference_tfidf --eval
```

### 单条推理

```bash
python -m inference.inference_tfidf
```

## 配置

在 `train/train_tfidf.py` 顶部 `TFIDF_CONFIG` 字典中修改：

| 参数 | 说明 |
|------|------|
| `max_features` | TF-IDF 最大特征词数 |
| `ngram_range` | n-gram 范围，(1,2) 为词 + 二元词组 |
| `min_df` | 词最少出现的文档数 |
| `C` | 正则化强度倒数，越小正则越强 |
| `max_iter` | 优化最大迭代次数 |

## 与 BiLSTM 对比

| | TF-IDF + LR | BiLSTM + Attention |
|---|---|---|
| 训练速度 | ~3 秒 | ~5 分钟 |
| 模型大小 | ~2 MB | ~8 MB |
| 文本表示 | 词频统计 | 词嵌入 + 上下文 |
| 可解释性 | 查看 LR 系数即可 | 需分析 Attention 权重 |
| 适用场景 | 快速 baseline、小数据 | 需要语义理解的任务 |
