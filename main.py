"""Rumor Detection — main entry point"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.inference import run_inference
from llm.llm import explain, LLMConfig

# ── 启动时检查 LLM API key ──────────────────────────────────────────
_llm_available = True
try:
    LLMConfig.from_env()   # 若 api_key 为空串则抛 ValueError
except ValueError as e:
    _llm_available = False
    print(f"[警告] {e}")
    print("[警告] LLM 解释功能已禁用，仅输出模型预测结果。\n")


def main():
    if not _llm_available:
        print("提示: 设置 LLM_API_KEY 环境变量后可启用 LLM 解释。\n")
    text = input("Enter text: ") 
    results = run_inference([text])
    for text, (label, conf, evidence) in zip([text], results):
        print(f"[{label}] conf={conf:.3f} evidence={evidence}")
        print(f"  text: {text[:80]}...")
        explanation = explain(text, label, evidence)
        if explanation:
            print(f"  explanation: {explanation}")
        print()


if __name__ == "__main__":
    main()
