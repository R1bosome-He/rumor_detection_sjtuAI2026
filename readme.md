# Rumor-Detection

基于 Twitter-RoBERTa 微调的谣言检测模型。模型权重托管在 [FENGYU21/rumor-detection](https://huggingface.co/FENGYU21/rumor-detection)，本地缺失时自动从 HuggingFace 下载。

## 快速开始

### 环境准备

推荐使用 Python 3.10 或更高版本。

```bash
# 克隆项目
git clone https://github.com/R1bosome-He/rumor_detection_sjtuAI2026.git
cd rumor_detection_sjtuAI2026

# （可选）创建虚拟环境
python -m venv venv
.\venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

### 模型推理

```bash
python main.py
```

单条文本测试：

```bash
python -m inference.inference
```

### 准确率验证

```bash
python inference/eval.py
```

### 模型训练

```bash
python train/train.py
```
