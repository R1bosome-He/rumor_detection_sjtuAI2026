import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.config import HF_REPO_ID, MAX_LENGTH, MODEL_DIR
from utils.data import preprocess_tweet


def load_threshold(model_dir=None):
    """Load classification threshold, with HF Hub fallback."""
    if model_dir is None:
        model_dir = MODEL_DIR
    threshold_path = Path(model_dir) / "threshold.json"

    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("threshold", 0.5))

    # Fallback: download from HuggingFace Hub
    print(f"Threshold file not found locally, downloading from {HF_REPO_ID} ...")
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename="threshold.json")
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("threshold", 0.5))
    except Exception:
        print("Warning: could not download threshold.json, using default 0.5")
        return 0.5


def predict_with_evidence(text, model, tokenizer, threshold=0.5, max_length=MAX_LENGTH):
    """Predict and return evidence words (RoBERTa BPE token merging via G-with-dot prefix)."""
    text = preprocess_tweet(text)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        rumor_prob = probs[1].item()
        label = int(rumor_prob >= threshold)
        confidence = rumor_prob if label == 1 else 1.0 - rumor_prob

        attentions = outputs.attentions[-1]
        cls_attn = attentions[:, :, 0, :]
        cls_attn_mean = cls_attn.mean(dim=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        stopwords = {
            "the", "a", "an", "and", "of", "to", "in", "for", "is", "on", "at", "by",
            "with", "from", "that", "this", "these", "those", "be", "are", "was", "were",
            "has", "have", "had", "does", "do", "did", "but", "or", "so", "not", "can",
            "will", "just", "like", "then", "now", "only", "very", "don", "should",
        }
        special_tokens = {"<s>", "</s>", "<pad>"}

        words = []
        word_token_indices = []
        for i, token in enumerate(tokens):
            if token in special_tokens:
                continue
            if token.startswith("\u0120"):
                words.append(token[1:])
                word_token_indices.append([i])
            else:
                if not words:
                    words.append(token)
                    word_token_indices.append([i])
                else:
                    words[-1] += token
                    word_token_indices[-1].append(i)

        valid_words = [
            (idxs, word) for idxs, word in zip(word_token_indices, words)
            if word.lower() not in stopwords and word.replace("'", "").isalpha()
        ]

        if not valid_words:
            return label, confidence, []

        scores = torch.stack([cls_attn_mean[0][idxs].mean() for idxs, _ in valid_words])
        top_k = min(5, len(valid_words))
        top_indices = torch.topk(scores, top_k).indices
        evidence = [valid_words[i][1] for i in top_indices]

    return label, confidence, evidence


class RumourDetectClass:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.threshold = load_threshold(self.model_dir)

        # Load model & tokenizer: local first, fall back to HuggingFace Hub
        has_config = (self.model_dir / "config.json").exists()
        has_weights = (
            (self.model_dir / "model.safetensors").exists()
            or (self.model_dir / "pytorch_model.bin").exists()
        )
        if has_config and has_weights:
            model_path = str(self.model_dir)
        else:
            if has_config and not has_weights:
                print("Local config found but weights missing, "
                      f"downloading from HuggingFace: {HF_REPO_ID}")
            else:
                print(f"Local model not found, loading from HuggingFace: {HF_REPO_ID}")
            model_path = HF_REPO_ID

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            attn_implementation="eager",
        )
        self.model.eval()

    def classify(self, text: str) -> int:
        label, _, _ = predict_with_evidence(text, self.model, self.tokenizer, self.threshold)
        return int(label)

    def classify_with_evidence(self, text: str):
        return predict_with_evidence(text, self.model, self.tokenizer, self.threshold)


def run_inference(texts):
    """Run inference on one or more texts.

    Parameters
    ----------
    texts : str or list of str
        A single text or a list of texts to classify.

    Returns
    -------
    list of (label, confidence, evidence)
    """
    if isinstance(texts, str):
        texts = [texts]
    clf = RumourDetectClass()
    results = [clf.classify_with_evidence(str(t)) for t in texts]
    return results


if __name__ == "__main__":
    test_texts = [
        "BREAKING: #Ferguson police chief just announced that officer Darren Wilson shot the unarmed teen, Michael Brown.",
        "Swiss museum confirms it will take on #Gurlitt collection",
    ]
    for label, conf, evidence in run_inference(test_texts):
        print(f"[{label}] conf={conf:.3f} evidence={evidence}")
