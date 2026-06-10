import os
import sys

# 自动定位项目根目录确保加载
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train.train import train
from inference.inference import predict_single, print_inference_result
from llm.llm import explain, LLMConfig

def main():
    print("=" * 70)
    print("  🏫 大学《人工智能导论》大作业 ——《可解释的谣言检测系统》")
    print("  🏆 核心算法基准: 【TF-IDF 向量化 + 逻辑回归 (Logistic Regression) 方案】")
    print("  🧠 认知解释层: 【上海交通大学 DeepSeek-Reasoner 事实核查研判分析器】")
    print("=" * 70)
    
    # 1. 确保模型已经训练完毕。如未训练，直接进行自动化初始化与极速训练。
    lr_file = os.path.join("saved_models", "lr_model.pkl")
    vec_file = os.path.join("saved_models", "tfidf_vectorizer.pkl")
    
    if not os.path.exists(lr_file) or not os.path.exists(vec_file):
        print("\n[System] 检测到首次运行，系统将自动基于内置的高仿真推文数据集为您进行一键极速训练！")
        train()
        print("-" * 70)
        
    presets = [
        "URGENT BREAKING: Shocking leak reveals that the president just passed away at 3 AM from a sudden cardiac arrest!",
        "Happy to announce we are officially launching our new community-driven programming tutorial series next Monday.",
        "CONFIRMED!! The military is arresting a top government official right now at the capital. Censorship acts activated!",
        "Studies in cognitive psychology show that maintaining a regular daily schedule in study helps cognitive retention."
    ]
    
    while True:
        print("\n请输入操作代码:")
        print("  [1] 交互式文本检测与可解释分析")
        print("  [2] 加载预置样例进行快速评估")
        print("  [3] 重新执行模型训练 (Re-train)")
        print("  [E] 退出程序")
        
        choice = input("您的选择: ").strip().lower()
        
        if choice == '1':
            user_text = input("\n请输入你想要检测的推文文本 (推荐英文): ").strip()
            if not user_text:
                print("  ⚠️ 输入不能为空，请重新选择。")
                continue
            
            # 运行模型推理
            res = predict_single(user_text)
            print_inference_result(res)
            
            if res.get("success", False):
                call_llm = input("\n是否连通交大 DeepSeek-Reasoner 接口进行大模型可解释分析？(y/n): ").strip().lower()
                if call_llm == 'y':
                    print("\n正在向上海交大 Models 平台 API 发送请求，运行 DeepSeek-Reasoner 推理，请稍候...")
                    explanation = explain(res["text"], res["label"], res["evidence"])
                    print("\n" + "=" * 30 + " [上海交大 DeepSeek 深度研判报告] " + "=" * 30)
                    print(explanation)
                    print("=" * 90)
                else:
                    print("  已跳过大模型解释生成。")
                    
        elif choice == '2':
            print("\n系统内置预置样例:")
            for idx, p in enumerate(presets):
                print(f"  [{idx + 1}] {p}")
            
            preset_choice = input("选择样例序号: ").strip()
            try:
                p_idx = int(preset_choice) - 1
                if 0 <= p_idx < len(presets):
                    target_text = presets[p_idx]
                    res = predict_single(target_text)
                    print_inference_result(res)
                    
                    call_llm = input("\n是否连通交大 DeepSeek-Reasoner 接口进行大模型可解释分析？(y/n): ").strip().lower()
                    if call_llm == 'y':
                        print("\n正在连通交大 API 启动深度可解释性核查...")
                        explanation = explain(res["text"], res["label"], res["evidence"])
                        print("\n" + "=" * 30 + " [上海交大 DeepSeek 深度研判报告] " + "=" * 30)
                        print(explanation)
                        print("=" * 90)
                else:
                    print("  ⚠️ 序号超出范围！")
            except ValueError:
                print("  ⚠️ 输入无效，请输入数字。")
                
        elif choice == '3':
            print("\n一键卸载当前模型并重新训练...")
            train()
            
        elif choice == 'e' or choice == 'q':
            print("\n感谢使用《可解释的谣言检测系统》! 祝您课程大作业顺利获得优秀 (A+) 等级！🎓")
            break
        else:
            print("  ⚠️ 无效指令，请重新输入。")

if __name__ == "__main__":
    main()