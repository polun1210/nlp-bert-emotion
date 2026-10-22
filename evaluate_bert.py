"""
evaluate_bert.py

用「從沒參與訓練、也沒參與挑選 checkpoint」的 test set，
誠實評估 output/ 裡的最佳模型。

輸出：

    終端機：每一類的 precision / recall / F1（classification report）
    output/confusion_matrix.png：混淆矩陣

為什麼不直接用 validation 的分數？

    train_bert.py 是「看 validation eval_loss 來挑最佳 checkpoint」，
    等於拿 validation 當考卷挑最會考的那一次，分數會偏樂觀。
    test set 才是真正的期末考。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import load_dataset
from matplotlib import font_manager
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =============================================================================
# 1. 設定
# =============================================================================

MODEL_DIR = Path("output")
TEST_FILE = Path("data/test.jsonl")

CONFUSION_MATRIX_FILE = MODEL_DIR / "confusion_matrix.png"

# matplotlib 預設字型沒有中文字，類別名稱會變成方框（□□□□）。
FONT_FILE = Path("fonts/NotoSansTC-Regular.otf")

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64


# =============================================================================
# 2. 輔助函式
# =============================================================================

def predict_labels(texts, tokenizer, model, device):
    """
    分批推論，回傳每一句的預測類別 id。
    """

    preds = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # no_grad：推論不需要計算 gradient，省記憶體也比較快。
        with torch.no_grad():
            logits = model(**inputs).logits

        preds.extend(logits.argmax(dim=-1).tolist())

    return np.array(preds)


def plot_confusion_matrix(labels, preds, label_names, path: Path):
    """
    混淆矩陣：

        列（row）    = 正確答案
        欄（column） = 模型預測

    對角線 = 答對的數量；對角線以外 = 「把 A 誤認成 B」的數量。
    看非對角線最大的格子，就知道模型最容易搞混哪兩種情緒。
    """

    if FONT_FILE.is_file():
        font_manager.fontManager.addfont(str(FONT_FILE))
        plt.rcParams["font.family"] = (
            font_manager.FontProperties(fname=str(FONT_FILE)).get_name()
        )

    matrix = confusion_matrix(
        labels, preds, labels=list(range(len(label_names)))
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=label_names,
    ).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)

    ax.set_xlabel("模型預測")
    ax.set_ylabel("正確答案")
    ax.set_title("Test Set Confusion Matrix")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 3. 主流程
# =============================================================================

def main():

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先執行 python train_bert.py"
        )

    if not TEST_FILE.is_file():
        raise FileNotFoundError(
            f"找不到 test data：{TEST_FILE.resolve()}\n"
            "請先執行 python prepare_data.py"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # id2label 存在模型 config 裡，
    # 所以評估程式不需要再讀 label_map.json。
    id2label = model.config.id2label
    label_names = [id2label[index] for index in sorted(id2label)]

    test_set = load_dataset("json", data_files=str(TEST_FILE))["train"]
    texts = test_set["sentences"]
    labels = np.array(test_set["labels"])

    preds = predict_labels(texts, tokenizer, model, device)

    print("=" * 72)
    print(f"Test set：{len(labels)} 筆")
    print(f"Accuracy：{accuracy_score(labels, preds):.4f}")
    print(f"Macro F1：{f1_score(labels, preds, average='macro'):.4f}")
    print("=" * 72)

    # precision：模型說是「開心」的句子裡，真的開心的比例（準不準）
    # recall   ：真的開心的句子裡，被模型抓出來的比例（漏不漏）
    # support  ：該類別在 test set 的實際筆數
    print(
        classification_report(
            labels,
            preds,
            labels=list(range(len(label_names))),
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    plot_confusion_matrix(labels, preds, label_names, CONFUSION_MATRIX_FILE)
    print(f"混淆矩陣：{CONFUSION_MATRIX_FILE.resolve()}")

    # 列出答錯的句子，方便人工檢查是模型的問題，還是標註本身就模稜兩可。
    wrong = np.flatnonzero(preds != labels)
    print("=" * 72)
    print(f"答錯 {len(wrong)} 筆，前 15 筆：")
    for index in wrong[:15]:
        print(
            f"  {texts[index]}\n"
            f"    正解={id2label[labels[index]]}  預測={id2label[preds[index]]}"
        )


if __name__ == "__main__":
    main()
