"""Data preprocessing and dataset"""
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.config import TRAIN_PATH, VAL_PATH, MAX_LENGTH

class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(str(self.texts[idx]), truncation=True, max_length=self.max_length)
        encoding["labels"] = int(self.labels[idx])
        return encoding

def preprocess_tweet(text: str) -> str:
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+", "http", text)
    return text

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    train_df["text"] = train_df["text"].apply(preprocess_tweet)
    val_df["text"] = val_df["text"].apply(preprocess_tweet)

    train_df = clean_train_data(train_df)
    return train_df, val_df

def clean_train_data(train_df):
    before = len(train_df)

    conflict_texts = (
        train_df.groupby("text")["label"]
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )
    conflict_rows = train_df["text"].isin(conflict_texts).sum()

    cleaned = train_df.loc[~train_df["text"].isin(conflict_texts)].copy()
    cleaned = cleaned.drop_duplicates(subset=["text", "label"], keep="first")

    removed = before - len(cleaned)
    print(
        f"训练集清洗：原始 {before} 条，移除 {removed} 条；"
        f"其中冲突重复文本 {len(conflict_texts)} 组/{conflict_rows} 条。"
    )
    return cleaned.reset_index(drop=True)
