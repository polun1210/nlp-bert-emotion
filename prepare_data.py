"""
prepare_data.py

把 Chinese_Multi-Emotion_Dialogue_Dataset 的 data.csv
轉成 BERT 多元分類（multi-class classification）訓練所需的 JSONL。

原始資料格式（CSV，有標題列）：

    text,emotion
    你要不要去吃午餐？,平淡語氣
    誒誒誒！我甄選上了！,開心語調

這支程式做的事情：

    data/data.csv
        ↓
    讀取 text / emotion，檢查格式
        ↓
    移除重複句子（避免同一句同時出現在 train 與 test）
        ↓
    文字 label → 整數 id（BERT 的 loss 只吃整數）
        ↓
    stratified split：80% train / 10% validation / 10% test
        ↓
    data/train.jsonl
    data/valid.jsonl
    data/test.jsonl
    data/label_map.json

和老師的二元分類範例相比，主要差異：

1. label 從 0 / 1 變成 8 種中文字串，需要建立 label ↔ id 的對照表。
2. 多切一份 test set。
   validation 會被拿來挑最佳 checkpoint，分數會偏樂觀；
   test 完全不參與訓練與挑選，才能誠實回報模型表現。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import json
from collections import Counter
from pathlib import Path

# pandas：
# 原始資料是標準 CSV（有標題列），直接用 read_csv 最方便。
import pandas as pd

from sklearn.model_selection import train_test_split


# =============================================================================
# 2. 資料設定
# =============================================================================

# 從 Hugging Face 下載的原始資料。
INPUT_FILE = Path("data/data.csv")

OUTPUT_DIR = Path("data")

# label ↔ id 對照表。
#
# train_bert.py 會讀它來設定 num_labels / id2label / label2id，
# 這樣類別定義只有一個來源（single source of truth），
# 不會出現兩支程式各自寫死、卻寫得不一樣的情況。
LABEL_MAP_FILE = OUTPUT_DIR / "label_map.json"

TEXT_COLUMN = "text"
LABEL_COLUMN = "emotion"

# 切分比例：
#
#     先切出 20% 當 holdout，
#     再把 holdout 對半分成 validation / test。
#
# 最後得到：
#     80% → training
#     10% → validation
#     10% → test
HOLDOUT_RATIO = 0.2
TEST_RATIO_IN_HOLDOUT = 0.5

SEED = 42


# =============================================================================
# 3. 輔助函式
# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """
    讀取 data.csv，做基本清理。

    回傳只含 text / emotion 兩欄的 DataFrame。
    """

    df = pd.read_csv(path, encoding="utf-8")

    missing_columns = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{path} 缺少欄位：{missing_columns}，實際欄位：{list(df.columns)}"
        )

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

    # 去掉前後空白，避免 "好開心 " 和 "好開心" 被當成兩句不同的話。
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    # -------------------------------------------------------------------------
    # 空值 / 空字串：沒有文字可學，直接略過
    # -------------------------------------------------------------------------
    is_empty = (df[TEXT_COLUMN] == "") | (df[TEXT_COLUMN] == "nan")
    if is_empty.any():
        print(f"已略過空白句子：{is_empty.sum()} 筆")
    df = df[~is_empty]

    # -------------------------------------------------------------------------
    # 重複句子：只保留第一筆
    # -------------------------------------------------------------------------
    #
    # 為什麼一定要去重？
    #
    # 如果同一句話出現兩次，隨機切分後很可能：
    #
    #     一筆在 train
    #     一筆在 test
    #
    # 模型等於「考試前看過考題」，test 分數會被灌水，
    # 這叫做資料洩漏（data leakage）。
    duplicated = df.duplicated(subset=[TEXT_COLUMN], keep="first")
    if duplicated.any():
        print(f"已移除重複句子：{duplicated.sum()} 筆")
    df = df[~duplicated]

    if df.empty:
        raise RuntimeError(f"{path} 沒有讀到任何有效資料。")

    return df.reset_index(drop=True)


def build_label_map(labels) -> dict:
    """
    建立 label（中文字串）→ id（整數）的對照表。

    為什麼要轉成整數？

    BERT 分類頭輸出的是 [batch_size, num_labels] 的 logits，
    CrossEntropyLoss 需要的正確答案是「第幾個位置」，也就是整數 index。

    為什麼要 sorted？

    set / unique 的順序不保證固定。
    排序後，每次執行 prepare_data.py 得到的對照表都一樣，
    已經訓練好的模型才不會因為重新跑前處理而「類別對錯號」。
    """

    return {
        label: index
        for index, label in enumerate(sorted(set(labels)))
    }


def save_jsonl(texts, label_ids, path: Path):
    """
    每一行格式：

        {"sentences": "我好開心", "labels": 5}

    欄位名稱沿用老師範例的 sentences / labels。
    labels 這個名字很重要：Trainer 會自動把名為 labels 的欄位當成正確答案。
    """

    with path.open("w", encoding="utf-8") as file:
        for text, label_id in zip(texts, label_ids):
            row = {
                "sentences": text,
                "labels": int(label_id),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_label_distribution(name, label_ids, id2label):
    """
    顯示每個類別的資料量與比例。
    """

    counts = Counter(label_ids)
    total = len(label_ids)

    print(f"{name}：{total} 筆")

    for label_id in sorted(counts):
        count = counts[label_id]
        print(
            f"  {label_id} {id2label[label_id]}: "
            f"{count:>4} 筆 ({count / total:.2%})"
        )


# =============================================================================
# 4. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：確認原始資料存在
    # -------------------------------------------------------------------------

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到原始資料：{INPUT_FILE.resolve()}\n"
            "請到 Hugging Face 下載 data.csv，放到 data/ 資料夾。"
        )

    # -------------------------------------------------------------------------
    # Step 2：讀取與清理
    # -------------------------------------------------------------------------

    df = load_data(INPUT_FILE)

    # -------------------------------------------------------------------------
    # Step 3：文字 label → 整數 id
    # -------------------------------------------------------------------------

    label2id = build_label_map(df[LABEL_COLUMN])
    id2label = {index: label for label, index in label2id.items()}

    texts = df[TEXT_COLUMN].tolist()
    label_ids = df[LABEL_COLUMN].map(label2id).tolist()

    print("=" * 72)
    print_label_distribution("全部資料", label_ids, id2label)

    # -------------------------------------------------------------------------
    # Step 4：stratified split（兩階段）
    # -------------------------------------------------------------------------

    # stratify 讓每一份資料的類別比例都和原始資料接近。
    #
    # 多元分類更需要這件事：
    # 例如「疑問語調」只有 386 筆，
    # 如果純亂數切分，test 裡的疑問語調可能特別少，
    # 該類別的分數就會很不穩定。
    (
        train_texts,
        holdout_texts,
        train_labels,
        holdout_labels,
    ) = train_test_split(
        texts,
        label_ids,
        test_size=HOLDOUT_RATIO,
        random_state=SEED,
        stratify=label_ids,
    )

    (
        valid_texts,
        test_texts,
        valid_labels,
        test_labels,
    ) = train_test_split(
        holdout_texts,
        holdout_labels,
        test_size=TEST_RATIO_IN_HOLDOUT,
        random_state=SEED,
        stratify=holdout_labels,
    )

    # -------------------------------------------------------------------------
    # Step 5：儲存 JSONL 與 label map
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "Training": (train_texts, train_labels, OUTPUT_DIR / "train.jsonl"),
        "Validation": (valid_texts, valid_labels, OUTPUT_DIR / "valid.jsonl"),
        "Test": (test_texts, test_labels, OUTPUT_DIR / "test.jsonl"),
    }

    for name, (split_texts, split_labels, path) in splits.items():
        save_jsonl(split_texts, split_labels, path)
        print_label_distribution(name, split_labels, id2label)
        print(f"  → {path}")

    with LABEL_MAP_FILE.open("w", encoding="utf-8") as file:
        json.dump(label2id, file, ensure_ascii=False, indent=2)

    print(f"Label map：{LABEL_MAP_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
