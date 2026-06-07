"""Rumor Detection — main entry point"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.inference import run_inference
from llm.llm import explain


def main():
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
