"""
infer_bert.py

載入 train_bert.py 儲存的最佳 BERT 八類情緒分類器，
預測幾個示範句子，輸出格式：

    我每天都能跟她一起上學，我好開心！ => 開心語調 (0.9876)

score（confidence）的意義：

    模型輸出 8 個 logits
        ↓ softmax
    8 個加總為 1 的機率
        ↓
    取最大的那一個 → 預測類別 + 它的機率（= score）

pipeline 預設只回傳第一名；
設定 top_k=None 可以看到全部 8 類的機率分布。
"""

from pathlib import Path

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# =============================================================================
# 1. 推論設定
# =============================================================================

MODEL_DIR = Path("output")

MAX_SEQ_LENGTH = 128

# 示範用的句子，涵蓋開心、悲傷、平淡等不同情緒。
TEST_TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
]

# 除了第一名，也顯示前幾名的機率，觀察模型在哪些類別之間猶豫。
SHOW_TOP_K = 3


# =============================================================================
# 2. 主流程
# =============================================================================

def main():

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先執行 python train_bert.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # device=0 → 第 0 張 CUDA GPU；device=-1 → CPU。
    device = 0 if torch.cuda.is_available() else -1

    print(
        "Inference device："
        + (torch.cuda.get_device_name(0) if device == 0 else "CPU")
    )

    # pipeline 會自動完成：
    #     tokenize → model forward → softmax → 用 config.id2label 轉成類別名稱
    #
    # 因為 train_bert.py 把 id2label 存進了 config.json，
    # 這裡直接就能拿到「開心語調」而不是「LABEL_5」。
    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # top_k=None：回傳每一句全部 8 類的分數（已依分數由高到低排序）。
    results = classifier(
        TEST_TEXTS,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        top_k=None,
    )

    # -------------------------------------------------------------------------
    # 輸出格式：句子 => 情緒 (機率)
    # -------------------------------------------------------------------------

    print("=" * 72)
    for text, scores in zip(TEST_TEXTS, results):
        best = scores[0]
        print(f"{text} => {best['label']} ({best['score']:.4f})")

    # -------------------------------------------------------------------------
    # 補充：前幾名的機率分布
    # -------------------------------------------------------------------------

    print("=" * 72)
    print(f"前 {SHOW_TOP_K} 名機率：")
    for text, scores in zip(TEST_TEXTS, results):
        top = "、".join(
            f"{item['label']} {item['score']:.4f}"
            for item in scores[:SHOW_TOP_K]
        )
        print(f"  {text}\n    {top}")
    print("=" * 72)


if __name__ == "__main__":
    main()
