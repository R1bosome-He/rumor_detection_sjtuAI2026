"""Rumor Detection — main entry point"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.inference import run_inference
from llm.llm import explain, set_api_key, _get_api_key

# ── 启动时检查 LLM API key ──────────────────────────────────────────
_llm_available = False

if __name__ == "__main__":
    if not _get_api_key():
        print("LLM API key 未设置。")
        print("  方式 1: 设置环境变量 LLM_API_KEY=your-key")
        print("  方式 2: 现在直接输入\n")
        key = input("请输入 API key (回车跳过则禁用 LLM 解释): ").strip()
        if key:
            set_api_key(key)
            _llm_available = True
            print("  API key 已设置。\n")
        else:
            print("[警告] LLM 解释功能已禁用。\n")
    else:
        _llm_available = True


def main():
    if not _llm_available:
        print("提示: 设置 LLM_API_KEY 后可启用 LLM 解释。\n")
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
