"""Training pipeline for Twitter-RoBERTa rumor detection"""
import inspect, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments, set_seed
from utils.config import BASE_DIR, CACHE_DIR, CONFIG, EARLY_STOPPING_PATIENCE, MAX_LENGTH, MODEL_DIR, MODEL_NAME, RESULTS_DIR
from utils.data import RumorDataset, preprocess_tweet, clean_train_data
from utils.metrics import compute_metrics, find_best_threshold
from train.trainer import WeightedTrainer

# JSONL 数据路径
TRAIN_JSON = BASE_DIR / "dataset" / "split" / "train.json"
VAL_JSON   = BASE_DIR / "dataset" / "split" / "val.json"


def _load_jsonl(path):
    """读取 JSONL，返回 DataFrame（兼容原有清洗和 Dataset 流程）"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if not obj:
                continue
            rows.append({"text": obj["text"],
                         "label": int(obj["label"]),
                         "id": obj.get("id", ""),
                         "event": obj.get("event", 0)})
    return pd.DataFrame(rows)


def load_data_from_json():
    """从 JSONL 加载并预处理，返回 (train_df, val_df)"""
    train_df = _load_jsonl(TRAIN_JSON)
    val_df = _load_jsonl(VAL_JSON)

    train_df["text"] = train_df["text"].apply(preprocess_tweet)
    val_df["text"] = val_df["text"].apply(preprocess_tweet)

    train_df = clean_train_data(train_df)
    print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条")
    return train_df, val_df

def build_training_args(config):
    output_dir = RESULTS_DIR / config["name"]
    args_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_strategy="epoch",
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        report_to="none",
        seed=config["seed"],
        data_seed=config["seed"],
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        push_to_hub=False,
    )

    training_args_sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_sig.parameters:
        args_kwargs["evaluation_strategy"] = "epoch"
    else:
        args_kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**args_kwargs)

def run_trial(config, tokenizer, train_dataset, val_dataset, data_collator, class_weights):
    print(f"\n{'='*10} Trial: {config['name']} {'='*10}")
    set_seed(config["seed"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        cache_dir=str(CACHE_DIR),
    )

    trainer = WeightedTrainer(
        model=model,
        args=build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()
    val_output = trainer.predict(val_dataset)
    threshold_info = find_best_threshold(val_output.predictions, val_output.label_ids)
    argmax_preds = np.argmax(val_output.predictions, axis=-1)
    argmax_acc = accuracy_score(val_output.label_ids, argmax_preds)
    argmax_f1 = f1_score(val_output.label_ids, argmax_preds, average="binary")

    print(
        f"Argmax: accuracy={argmax_acc:.4f}, f1={argmax_f1:.4f} | "
        f"Best threshold={threshold_info['threshold']:.4f}, "
        f"accuracy={threshold_info['accuracy']:.4f}, f1={threshold_info['f1']:.4f}"
    )

    return {
        "config": config,
        "trainer": trainer,
        "val_output": val_output,
        "threshold_info": threshold_info,
        "argmax_accuracy": float(argmax_acc),
        "argmax_f1": float(argmax_f1),
    }

def save_best_model(result, tokenizer):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result["trainer"].model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    ti = result["threshold_info"]
    summary = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "best_config": result["config"],
        "argmax_accuracy": result["argmax_accuracy"],
        "argmax_f1": result["argmax_f1"],
        "threshold": ti["threshold"],
        "threshold_accuracy": ti["accuracy"],
        "threshold_f1": ti["f1"],
    }
    
    with open(MODEL_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nModel saved to {MODEL_DIR}")

def main():
    train_df, val_df = load_data_from_json()
    print(f"Loading model {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
    train_dataset = RumorDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = RumorDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    labels_train = train_df["label"].to_numpy()
    class_counts = np.bincount(labels_train, minlength=2)
    class_weights = (class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))).astype(np.float32)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Labels: {dict(enumerate(class_counts.tolist()))}, weights={class_weights.tolist()}")
    result = run_trial(CONFIG, tokenizer, train_dataset, val_dataset, data_collator, class_weights)
    save_best_model(result, tokenizer)
    print("\n=== Final eval ===")
    labels = result["val_output"].label_ids
    preds = result["threshold_info"]["preds"]
    print(classification_report(labels, preds, target_names=["non-rumor", "rumor"]))
    print("CM:", confusion_matrix(labels, preds).tolist())

if __name__ == "__main__":
    main()
