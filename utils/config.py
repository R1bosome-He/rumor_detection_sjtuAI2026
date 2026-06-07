"""训练与推理配置常量"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent                      

TRAIN_PATH = BASE_DIR / "dataset" / "split" / "train.csv"
VAL_PATH   = BASE_DIR / "dataset" / "split" / "val.csv"

MODEL_NAME = "cardiffnlp/twitter-roberta-base"

MODEL_DIR   = BASE_DIR / "model" / "twitter-roberta"
HF_REPO_ID  = "FENGYU21/rumor-detection"          # HuggingFace 仓库
CACHE_DIR   = BASE_DIR / "model" / "cache"
RESULTS_DIR = BASE_DIR / "train" / "results"

MAX_LENGTH = 128
EARLY_STOPPING_PATIENCE = 2

CONFIG = {
    "name": "lr2e-5_seed42",
    "learning_rate": 2e-5,
    "num_train_epochs": 4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "seed": 42,
}
