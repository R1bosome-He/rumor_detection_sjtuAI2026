import os
import sys

# 保证能正确导入上级目录中的 model 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lr_model import load_model, get_word_contributions

def predict_single(text):
    """
    单条文本推理逻辑，输出预测分类、置信度以及核心机制证据词。
    """
    try:
        model, vectorizer = load_model()
    except FileNotFoundError as e:
        print(f"[Inference Error] {str(e)}")
        # 抛出异常或提示进行训练
        return {
            "success": False,
            "error": "Model not trained yet."
        }

    # 1. 预测概率
    tf_idf_vector = vectorizer.transform([text])
    probabilities = model.predict_proba(tf_idf_vector)[0]  # [prob_class_0, prob_class_1]
    
    # 2. 预测标签
    predicted_label = int(probabilities[1] >= 0.5)
    confidence = float(probabilities[predicted_label])
    
    # 3. 提取解释性证据词 (Evidence Extraction)
    contributions = get_word_contributions(text, model, vectorizer)
    
    # 根据预测结果筛选证据词
    if predicted_label == 1:
        # 如果预测为谣言：倾向于寻找推动预测往 1 走的、正贡献度最大的前 5 个词
        sorted_contributions = sorted(
            contributions, 
            key=lambda x: x["contribution"], 
            reverse=True
        )
        # 过滤只保留正贡献较大 (或全部，如果没有正贡献也取最顶部的)
        evidences = sorted_contributions[:5]
    else:
        # 如果预测为非谣言：倾向于寻找推动预测往 0 走的、负贡献绝对值最大（也即贡献度最负）的前 5 个词
        sorted_contributions = sorted(
            contributions, 
            key=lambda x: x["contribution"], 
            reverse=False  # 由小到大排序，最负的在前面
        )
        evidences = sorted_contributions[:5]
        
    return {
        "success": True,
        "text": text,
        "label": predicted_label,
        "confidence": confidence,
        "evidence": evidences
    }

def print_inference_result(res):
    """
    格式化打印推理结果给用户
    """
    if not res.get("success", False):
        print(f"推理失败: {res.get('error')}")
        return
        
    label_desc = "🚨 谣言 (Rumor)" if res["label"] == 1 else "✅ 非谣言 (Non-Rumor)"
    print("-" * 60)
    print(f"【输入文本】: {res['text']}")
    print(f"【检测结论】: {label_desc}  (预测置信度: {res['confidence']:.2%})")
    print("【可解释证据词 (Top Evidence Words)】:")
    
    if not res["evidence"]:
        print("  ⚠️ 没有检出被激活/具备显著贡献的单词。")
    else:
        for idx, item in enumerate(res["evidence"]):
            direct_desc = "→ 谣言" if item["contribution"] > 0 else "→ 非谣言"
            print(f"  {idx + 1}. \"{item['word']}\": TF-IDF={item['tfidf']:.4f} * Beta={item['weight']:.4f} -> 贡献={item['contribution']:.4f} ({direct_desc})")
    print("-" * 60)

if __name__ == "__main__":
    test_text_rumor = "URGENT BREAKING NEWS: Shocking leak revealing sudden arrest of top officials tonight!"
    test_text_normal = "We announced next month we will host an amazing technical seminar on deep learning."
    
    print("\n[Inference] 测试谣言预测:")
    res_rumor = predict_single(test_text_rumor)
    print_inference_result(res_rumor)
    
    print("\n[Inference] 测试非谣言预测:")
    res_normal = predict_single(test_text_normal)
    print_inference_result(res_normal)