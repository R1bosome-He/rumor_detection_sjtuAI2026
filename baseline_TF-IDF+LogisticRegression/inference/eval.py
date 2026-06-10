import os
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 保证能正确导入上级目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lr_model import load_model

def evaluate():
    print("====== [Evaluation] 正在评测传统经典机器学习基准模型性能 ======")
    
    val_path = "dataset/split/val.csv"
    if not os.path.exists(val_path):
        print(f"[Error] 未找到验证集 CSV 文件: {val_path}")
        print("请先通过运行 train/train.py 进行训练并自动生成临时验证集！")
        return
        
    # 1. 加载验证数据集
    df_val = pd.read_csv(val_path, encoding='utf-8')
    print(f"[Evaluation] 成功读取验证语料，样本量: {len(df_val)}")
    
    if 'text' not in df_val.columns or 'label' not in df_val.columns:
        print("[Error] 验证集数据格式不规范，必须包含 'text' 和 'label' 两列！")
        return
        
    df_val = df_val.dropna(subset=['text', 'label'])
    X_val = df_val['text'].astype(str)
    y_true = df_val['label'].astype(int)
    
    # 2. 加载已经训练好的模型和向量化器
    try:
        model, vectorizer = load_model()
    except FileNotFoundError as e:
        print(f"[Error] 模型加载失败: {str(e)}")
        print("请先行在终端运行: python train/train.py")
        return
        
    # 3. 向量化与批量预测
    X_val_vec = vectorizer.transform(X_val)
    y_pred = model.predict(X_val_vec)
    
    # 4. 计算学术界通用的二分类各项评测指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("-" * 50)
    print(f"【分类准确率 (Accuracy)】:   {accuracy:.4f}")
    print(f"【分类精确率 (Precision)】:  {precision:.4f}")
    print(f"【分类召回率 (Recall)】:     {recall:.4f}")
    print(f"【F1-Score (综合调和平均)】:   {f1:.4f}")
    print("-" * 50)
    
    print("\n【分类结果详细报告 (Classification Report)】:")
    print(classification_report(y_true, y_pred, target_names=["Non-Rumor (0)", "Rumor (1)"], zero_division=0))
    print("====== [Evaluation] 评测结束 ======")

if __name__ == "__main__":
    evaluate()