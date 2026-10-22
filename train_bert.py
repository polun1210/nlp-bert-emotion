"""
train_bert.py

使用 google-bert/bert-base-chinese 做繁體中文句子「八類情緒」多元分類 Fine-tuning。

資料來源（由 prepare_data.py 產生）：

    data/train.jsonl
    data/valid.jsonl
    data/label_map.json

整體流程：

    JSONL
        ↓
    Hugging Face Dataset
        ↓
    Tokenizer（truncation，不先 padding）
        ↓
    DataCollatorWithPadding（每個 batch 動態 padding）
        ↓
    BERT encoder
        ↓
    [CLS] 向量 → Linear(768 → 8)   ← Sequence Classification Head
        ↓
    logits [batch_size, 8]
        ↓
    Cross-Entropy Loss（softmax + 負對數似然）
        ↓
    Backward → Optimizer Step
        ↓
    每 EVAL_STEPS 做一次 Validation
        ↓
    Early Stopping / 挑 eval_loss 最低的 checkpoint
        ↓
    output/

二元 → 多元，真正需要改的地方：

1. num_labels：2 → 8。
   AutoModelForSequenceClassification 會依照 num_labels
   建立對應大小的分類頭，輸出 8 個 logits。

2. id2label / label2id：從 label_map.json 讀取。
   存進模型 config 之後，pipeline 推論時就能直接顯示「開心語調」，
   而不是 LABEL_5。

3. F1 的 average：binary → macro。
   average="binary" 只適用於兩類，八類會直接報錯。
   macro = 每一類各自算 F1 再平均，每一類權重相同，
   樣本少的類別（例如疑問語調）表現差時也會反映在分數上。

Loss 不需要改：
當 num_labels > 1 且 labels 是整數時，
BertForSequenceClassification 會自動使用 CrossEntropyLoss，
二元與多元用的是同一套 loss。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import json
from pathlib import Path

import numpy as np
import torch
import transformers

from datasets import load_dataset

from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =============================================================================
# 2. 模型與路徑設定
# =============================================================================

MODEL_NAME = "google-bert/bert-base-chinese"

DATA_DIR = Path("data")

TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"
LABEL_MAP_FILE = DATA_DIR / "label_map.json"

# Trainer 的 checkpoint 暫存區。
#
# 訓練中會產生 checkpoint-* 資料夾（每個約 1.2 GB，含 optimizer 狀態），
# 不需要放進 repo，所以和最終模型分開放。
CHECKPOINT_DIR = Path("checkpoints")

# 最終要交的最佳模型（只含推論需要的檔案）。
OUTPUT_DIR = Path("output")


# =============================================================================
# 3. 模型設定
# =============================================================================

# 資料最長的句子只有 92 個字，平均約 20 字。
#
# bert-base-chinese 中文大致是一個字一個 token，
# 加上 [CLS] / [SEP] 也遠低於 128。
#
# 設 128 可以涵蓋全部資料，
# 推論時遇到特別長的輸入也會被截斷，不會超過模型上限 512。
MAX_SEQ_LENGTH = 128


# =============================================================================
# 4. 訓練設定
# =============================================================================

SEED = 42

# 最多訓練 5 個 epochs。
#
# 實際上常常不會跑完，
# 因為 EarlyStoppingCallback 會在 eval_loss 不再下降時提前停止。
NUM_TRAIN_EPOCHS = 5

PER_DEVICE_TRAIN_BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH_SIZE = 64

# 資料量小、GPU 記憶體足夠（RTX 5070 Ti 16GB），
# batch 32 可以直接放進 GPU，不需要 gradient accumulation。
GRADIENT_ACCUMULATION_STEPS = 1

# BERT 論文建議 fine-tuning 使用 2e-5、3e-5、5e-5。
#
# 1e-4 對 BERT 全模型微調偏大：
# 在只有三千多筆的小資料集上，
# 太大的 learning rate 容易破壞預訓練學到的語言知識
# （catastrophic forgetting），訓練曲線也比較抖。
LEARNING_RATE = 3e-5

# 每個 epoch 約：
#
#     3323 / 32 ≈ 104 optimizer steps
#
# 5 epochs 約 520 steps。
#
# 前 50 steps（約 10%）做 linear warmup：
# 一開始分類頭是隨機初始化，gradient 很大，
# 先用很小的 learning rate 慢慢升上去，避免一開始就把 BERT 權重打亂。
WARMUP_STEPS = 50

WEIGHT_DECAY = 0.01

LR_SCHEDULER_TYPE = "linear"

# 每 25 個 optimizer steps 做一次 validation，
# 520 steps 約有 20 個觀察點，early stopping 才判斷得準。
EVAL_STEPS = 25
SAVE_STEPS = 25

SAVE_TOTAL_LIMIT = 2

# 連續 3 次 validation（= 75 steps，約 0.7 個 epoch）
# eval_loss 都沒有創新低，就提前停止。
EARLY_STOPPING_PATIENCE = 3

LOGGING_STEPS = 10


# =============================================================================
# 5. Mixed Precision
# =============================================================================

# RTX 50 系列（Blackwell）原生支援 BF16。
#
# BF16 和 FP16 都是 16-bit，速度差不多，
# 但 BF16 的指數位數和 FP32 一樣，數值範圍大，
# 不會 overflow，也不需要 loss scaling，訓練比較穩定。
#
# 沒有 GPU 或 GPU 不支援 BF16 時，main() 會自動關閉。
USE_BF16 = True


# =============================================================================
# 6. 輔助函式
# =============================================================================

def print_gpu_info():
    """
    顯示目前 PyTorch 是否看到 GPU。
    """

    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")
    print(f"CUDA available：{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"CUDA capability：{torch.cuda.get_device_capability(0)}")
        print(f"PyTorch CUDA：{torch.version.cuda}")

    print("=" * 72)


def load_label_map(path: Path):
    """
    讀取 prepare_data.py 產生的 label_map.json。

    回傳：
        label2id : {"厭惡語調": 0, ...}
        id2label : {0: "厭惡語調", ...}
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 label map：{path.resolve()}\n"
            "請先執行 python prepare_data.py"
        )

    with path.open("r", encoding="utf-8") as file:
        label2id = json.load(file)

    id2label = {index: label for label, index in label2id.items()}

    return label2id, id2label


def compute_metrics(eval_pred):
    """
    Hugging Face Trainer 每次 evaluation 後呼叫。

    eval_pred.predictions shape：

        [num_examples, 8]

    每一列是 8 個類別的 logits，例如：

        [-1.2, 0.3, -0.8, -2.0, 0.1, 4.5, -0.6, 1.1]
                                    ↑
                              最大值在 index 5
                              → 預測為 5 = 開心語調

    argmax 取最大值的位置即可，
    不需要先做 softmax（softmax 不會改變大小順序）。
    """

    labels = eval_pred.label_ids
    predictions = eval_pred.predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),

        # macro：每類 F1 等權平均 → 看「每一類都學得好不好」
        "f1_macro": f1_score(
            labels, preds, average="macro", zero_division=0
        ),

        # weighted：依各類樣本數加權 → 比較接近整體 accuracy 的意義
        "f1_weighted": f1_score(
            labels, preds, average="weighted", zero_division=0
        ),
    }


# =============================================================================
# 7. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：GPU 資訊
    # -------------------------------------------------------------------------

    print_gpu_info()

    # -------------------------------------------------------------------------
    # Step 2：確認資料與 label map
    # -------------------------------------------------------------------------

    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE

    for path in (train_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到資料：{path.resolve()}\n"
                "請先執行 python prepare_data.py"
            )

    label2id, id2label = load_label_map(LABEL_MAP_FILE)
    num_labels = len(label2id)

    print(f"類別數：{num_labels}")
    for index in sorted(id2label):
        print(f"  {index} → {id2label[index]}")

    # -------------------------------------------------------------------------
    # Step 3：讀取 JSONL
    # -------------------------------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(valid_path),
        },
    )

    print(dataset)

    # -------------------------------------------------------------------------
    # Step 4：載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -------------------------------------------------------------------------
    # Step 5：Tokenization
    # -------------------------------------------------------------------------

    def tokenize_batch(batch):
        """
        只做 tokenization + truncation，padding 交給 DataCollatorWithPadding。
        """

        return tokenizer(
            batch["sentences"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["sentences"],
    )

    # -------------------------------------------------------------------------
    # Step 6：Dynamic Padding
    # -------------------------------------------------------------------------

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------------------------------------------------------------
    # Step 7：建立 BERT sequence classifier（8 類）
    # -------------------------------------------------------------------------

    # 這一行就是二元 → 多元最核心的改變。
    #
    # num_labels=8 時，分類頭是：
    #
    #     Linear(768 → 8)
    #
    # 這一層的權重是隨機初始化的（BERT 預訓練時沒有它），
    # 所以載入時會看到類似：
    #
    #     Some weights of BertForSequenceClassification were not initialized
    #     ... ['classifier.bias', 'classifier.weight']
    #
    # 這是正常的警告，代表「分類頭需要靠 fine-tuning 學」。
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # -------------------------------------------------------------------------
    # Step 8：TrainingArguments
    # -------------------------------------------------------------------------

    bf16_enabled = (
        USE_BF16
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),

        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        logging_steps=LOGGING_STEPS,

        # 訓練結束（跑完或 early stop）後，
        # 把 eval_loss 最低的 checkpoint 載回 trainer.model。
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=bf16_enabled,

        seed=SEED,
        data_seed=SEED,

        report_to="none",
    )

    # -------------------------------------------------------------------------
    # Step 9：建立 Trainer
    # -------------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    # -------------------------------------------------------------------------
    # Step 10：開始 fine-tuning
    # -------------------------------------------------------------------------

    trainer.train()

    # -------------------------------------------------------------------------
    # Step 11：顯示最佳 checkpoint 與它在 validation 上的表現
    # -------------------------------------------------------------------------

    print(f"最佳 checkpoint：{trainer.state.best_model_checkpoint}")
    print(f"最佳 eval_loss：{trainer.state.best_metric}")

    # 此時 trainer.model 已經是最佳 checkpoint，
    # 再跑一次 evaluate 可以看到它完整的 accuracy / F1。
    best_metrics = trainer.evaluate()
    print("最佳模型 validation 指標：")
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")

    # -------------------------------------------------------------------------
    # Step 12：儲存最佳模型（放進 repo 的 output/）
    # -------------------------------------------------------------------------

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # trainer_state.json 存在 CHECKPOINT_DIR，
    # 這裡複製一份到 OUTPUT_DIR，讓 plot_bert_metrics.py 讀得到。
    trainer.save_state()
    (OUTPUT_DIR / "trainer_state.json").write_text(
        (CHECKPOINT_DIR / "trainer_state.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(f"Final model directory：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
