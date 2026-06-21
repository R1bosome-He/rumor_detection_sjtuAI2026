import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sys

# 保证能正确导入上级目录中的 model 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lr_model import save_model

def construct_dummy_datasets():
    """
    如果用户本地没有数据集，为了实现零门槛直接运行而构造的逼真谣言与非谣言数据集。
    """
    train_data = [
        # 谣言 (Label 1)
        {"text": "URGENT BREAKING: Shocking leak reveals that the president just passed away at 3 AM from a sudden cardiac arrest!", "label": 1},
        {"text": "CONFIRMED!! The military is arresting a top government official right now at the capital. Details are being censored!", "label": 1},
        {"text": "This is a must-share! Secret documents prove that the government is planning a complete lock-down is starting tomorrow morning. Stock up on food!", "label": 1},
        {"text": "Alert: A dangerous chemical gas leak has been reported in the busy city center. Evacuate immediately! Authorities are hiding this!", "label": 1},
        {"text": "Shocking investigation showing that the global epidemic was completely fabricated in a dark secret lab. Unbelievable conspiratorial proof!", "label": 1},
        {"text": "URGENT: Banks are shutting down their systems at midnight. Move all physical money out of your accounts now! Censorship active!", "label": 1},
        {"text": "Breaking news: Huge alien spacecraft has landed in a remote region. NASA official leaks photos that are being wiped from social feeds!", "label": 1},
        
        # 非谣言 (Label 0)
        {"text": "We are thrilled to announced that our core research team has successfully published their latest study on quantum physics today.", "text": "Happy to announce we are officially launching our new community-driven programming tutorial series next Monday.", "label": 0},
        {"text": "According to the national weather report, heavy rainfall is scheduled across the state. Please carry an umbrella.", "label": 0},
        {"text": "Our weekly company alignment meeting has been complete. Team goals for the next quarter have been finalized.", "label": 0},
        {"text": "Review of the new smartphone: Clean display, versatile battery life, and excellent camera specs for a standard price.", "label": 0},
        {"text": "The local city council announced a new public library expansion project starting next month. Citizens are welcome to check details.", "label": 0},
        {"text": "Studies in cognitive psychology show that maintaining a regular daily schedule in study helps cognitive retention.", "label": 0},
        {"text": "The university is hosting a guest seminar on natural language processing. Check the schedule link if you wish to participate.", "label": 0}
    ]
    
    val_data = [
        # 验证集谣言
        {"text": "ALERT: Shocking leak showing that a massive emergency has been declared at the central nuclear power station!", "label": 1},
        {"text": "BREAKING: Secret documents reveal a sudden government censorship act starting midnight. Share this before it gets deleted!", "label": 1},
        # 验证集非谣言
        {"text": "The latest paper on deep learning models is published in the open archive today. Read it for structural details.", "label": 0},
        {"text": "We had an amazing team lunch today, celebrating the official launch of our technical project.", "label": 0}
    ]
    
    os.makedirs("dataset/split", exist_ok=True)
    train_path = "dataset/split/train.csv"
    val_path = "dataset/split/val.csv"
    
    if not os.path.exists(train_path):
        pd.DataFrame(train_data).to_csv(train_path, index=False, encoding='utf-8')
        print(f"[Dataset] 未检测到训练集，已自动初始化逼真的训练集于: {train_path}")
    if not os.path.exists(val_path):
        pd.DataFrame(val_data).to_csv(val_path, index=False, encoding='utf-8')
        print(f"[Dataset] 未检测到验证集，已自动初始化逼真的验证集于: {val_path}")

def train():
    print("====== [Train] 开始一键训练传统经典机器学习基准模型 ======")
    
    # 构建数据备份（如果不存在）
    construct_dummy_datasets()
    
    train_path = "dataset/split/train.csv"
    if not os.path.exists(train_path):
        print(f"[Error] 数据集路径不存在: {train_path}")
        return
        
    # 1. 加载数据集
    df_train = pd.read_csv(train_path, encoding='utf-8')
    print(f"[Train] 成功读取训练语料，样本量: {len(df_train)}")
    
    # 校验关键列
    if 'text' not in df_train.columns or 'label' not in df_train.columns:
        print("[Error] 训练数据格式不规范，必须包含 'text' 和 'label' 两列！")
        return
        
    df_train = df_train.dropna(subset=['text', 'label'])
    X_train = df_train['text'].astype(str)
    y_train = df_train['label'].astype(int)
    
    # 2. TF-IDF 特征工程向量化
    # 采用适合 Twitter 语料的参数设置：保留英文停用词过滤，提取1-2元语法词
    print("[Train] 正在构建 TF-IDF 特征向量...")
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        stop_words='english', 
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    print(f"[Train] 构建完成。特征维度/词表大小: {X_train_vec.shape[1]}")
    
    # 3. 逻辑回归分类器训练
    print("[Train] 正在训练逻辑回归分类器...")
    model = LogisticRegression(
        C=1.0, 
        class_weight='balanced', 
        random_state=42, 
        solver='liblinear'
    )
    model.fit(X_train_vec, y_train)
    print("[Train] 逻辑回归分类器训练完成！")
    
    # 4. 保存模型与特征矩阵
    save_model(model, vectorizer)
    print("====== [Train] 训练流程全部顺利结束！ ======")

if __name__ == "__main__":
    train()