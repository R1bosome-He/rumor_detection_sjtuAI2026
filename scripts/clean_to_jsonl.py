"""
把原始数据清洗成一行一个 JSON 对象的文件。

输出格式示例：
{"id":"536824152345678912","text":"Swiss museum confirms it will take on #Gurlitt collection","label":1,"event":0}

清洗规则：
1. 只保留 id、text、label、event 四个字段。
2. id 强制保存为字符串。
3. text 做基础清洗：替换用户名、替换链接、压缩多余空白。
4. label 只能是 0 或 1。
5. event 转为整数。
6. 删除 text 相同但 label 不一致的矛盾样本。
7. 删除重复样本。
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def preprocess_tweet(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+", "http", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_raw_data(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path, dtype={"id": str})
    if suffix == ".tsv":
        return pd.read_csv(input_path, sep="\t", dtype={"id": str})
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(input_path, dtype={"id": str})
    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if suffix == ".json":
        text = input_path.read_text(encoding="utf-8").strip()
        if not text:
            return pd.DataFrame()

        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict) and "data" in obj:
                return pd.DataFrame(obj["data"])
            if isinstance(obj, dict):
                return pd.DataFrame([obj])
        except json.JSONDecodeError:
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return pd.DataFrame(rows)

    raise ValueError(f"不支持的文件格式：{suffix}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["id", "text", "label", "event"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"原始数据缺少字段：{missing}，当前字段为：{list(df.columns)}"
        )

    before = len(df)

    cleaned = df[required_cols].copy()

    cleaned["id"] = cleaned["id"].astype(str).str.strip()
    cleaned["text"] = cleaned["text"].apply(preprocess_tweet)
    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="coerce")
    cleaned["event"] = pd.to_numeric(cleaned["event"], errors="coerce")

    cleaned = cleaned.dropna(subset=["id", "text", "label", "event"])
    cleaned = cleaned[cleaned["id"] != ""]
    cleaned = cleaned[cleaned["text"] != ""]
    cleaned = cleaned[cleaned["label"].isin([0, 1])]

    cleaned["label"] = cleaned["label"].astype(int)
    cleaned["event"] = cleaned["event"].astype(int)

    # 删除矛盾样本：
    # 如果同一个 text 同时对应 label=0 和 label=1，则整组删除。
    conflict_texts = (
        cleaned.groupby("text")["label"]
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )

    conflict_rows = cleaned["text"].isin(conflict_texts).sum()
    cleaned = cleaned.loc[~cleaned["text"].isin(conflict_texts)].copy()

    # 删除重复样本。
    cleaned = cleaned.drop_duplicates(subset=["id"], keep="first")
    cleaned = cleaned.drop_duplicates(subset=["text", "label", "event"], keep="first")

    after = len(cleaned)

    print(f"原始样本数：{before}")
    print(f"矛盾文本组数：{len(conflict_texts)}")
    print(f"矛盾样本行数：{conflict_rows}")
    print(f"清洗后样本数：{after}")
    print(f"总删除样本数：{before - after}")

    return cleaned.reset_index(drop=True)


def save_as_json_lines(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "id": str(row["id"]),
                "text": str(row["text"]),
                "label": int(row["label"]),
                "event": int(row["event"]),
            }
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始数据路径")
    parser.add_argument("--output", required=True, help="清洗后输出路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_raw_data(input_path)
    cleaned = clean_dataframe(df)
    save_as_json_lines(cleaned, output_path)

    print(f"已保存到：{output_path}")


if __name__ == "__main__":
    main()