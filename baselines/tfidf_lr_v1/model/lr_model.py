import os
import pickle
import numpy as np

def get_model_paths():
    """
    获取默认的模型和向量化器保存路径
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "lr_model.pkl")
    vec_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
    return model_path, vec_path

def save_model(model, vectorizer, model_path=None, vec_path=None):
    """
    保存逻辑回归模型和TF-IDF向量化器
    """
    default_m_path, default_v_path = get_model_paths()
    m_path = model_path or default_m_path
    v_path = vec_path or default_v_path
    
    with open(m_path, 'wb') as f:
        pickle.dump(model, f)
    with open(v_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[Model] 成功保存逻辑回归模型至: {m_path}")
    print(f"[Model] 成功保存TF-IDF向量化器至: {v_path}")

def load_model(model_path=None, vec_path=None):
    """
    加载逻辑回归模型和TF-IDF向量化器
    """
    default_m_path, default_v_path = get_model_paths()
    m_path = model_path or default_m_path
    v_path = vec_path or default_v_path
    
    if not os.path.exists(m_path) or not os.path.exists(v_path):
        raise FileNotFoundError(
            f"未找到已存模型或向量化器文件。请先运行 train/train.py 进行训练！\n 路径检测: \n- {m_path}\n- {v_path}"
        )
        
    with open(m_path, 'rb') as f:
        model = pickle.load(f)
    with open(v_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def get_word_contributions(text, model, vectorizer):
    """
    核心可解释性算法 (Evidence 提取机制)：
    计算输入文本中被 TF-IDF 激活的每个单词的贡献度。
    物理意义：贡献度 = 单词的 TF-IDF 值 * 该词在逻辑回归模型中的权重系数 (Beta Coefficient)
    对于谣言预测 (Label 1)，正值代表词语在向谣言方向推进；负值向非谣言方向推进。
    """
    # 提取特征词表与模型权重
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]  # 对于二分类，coef_ 的 Shape 是 (1, n_features)
    
    # 向量化单条文本
    tf_idf_vector = vectorizer.transform([text])
    
    # 找出文本中 TF-IDF 激活的特征索引和对应的 TF-IDF 值
    # tf_idf_vector 是 csr_matrix 稀疏矩阵
    non_zero_indices = tf_idf_vector.nonzero()[1]
    
    contributions = []
    for idx in non_zero_indices:
        word = feature_names[idx]
        tf_idf_val = tf_idf_vector[0, idx]
        weight = coef[idx]
        contribution = tf_idf_val * weight
        
        contributions.append({
            "word": word,
            "tfidf": float(tf_idf_val),
            "weight": float(weight),
            "contribution": float(contribution)
        })
        
    return contributions