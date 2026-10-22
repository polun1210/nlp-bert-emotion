"""
plot_bert_metrics.py

讀取 train_bert.py 產生的 output/trainer_state.json，輸出：

    output/loss_curve.png     Training / Validation loss
    output/metric_curve.png   Validation accuracy / macro F1

並在圖上標出最佳 checkpoint（eval_loss 最低）的位置。

使用 Matplotlib Agg backend 直接存成 PNG，不依賴 plt.show()，
沒有桌面 GUI 的環境（例如 Colab remote terminal）也能執行。
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =============================================================================
# 1. 路徑設定
# =============================================================================

OUTPUT_DIR = Path("output")

STATE_FILE = OUTPUT_DIR / "trainer_state.json"

LOSS_PLOT_FILE = OUTPUT_DIR / "loss_curve.png"

METRIC_PLOT_FILE = OUTPUT_DIR / "metric_curve.png"

# 要畫在 metric 圖上的 validation 指標（compute_metrics 回傳的 key 加上 eval_ 前綴）。
METRIC_KEYS = {
    "eval_accuracy": "Validation Accuracy",
    "eval_f1_macro": "Validation Macro F1",
}


# =============================================================================
# 2. 輔助函式
# =============================================================================

def extract_points(logs, key):
    """
    從 log_history 取出 (step, value) 的序列。

    只取到訓練結束（含 train_runtime 的那筆）為止。
    train_bert.py 在訓練後會對最佳模型再跑一次 evaluate()，
    那筆紀錄的 step 是最後一步、模型卻是最佳 checkpoint，畫進去會誤導。
    """

    points = []

    for item in logs:
        if "train_runtime" in item:
            break
        if "step" in item and key in item:
            points.append((item["step"], item[key]))

    return points


def find_best_step(state):
    """
    best_model_checkpoint 例如 "checkpoints\\checkpoint-325"，
    取出最後的數字 325 當作最佳 step。
    """

    checkpoint = state.get("best_model_checkpoint")
    if not checkpoint:
        return None

    return int(Path(checkpoint).name.split("-")[-1])


def mark_best_step(best_step):
    """
    在目前的圖上畫一條垂直虛線，標示最佳 checkpoint。
    """

    if best_step is not None:
        plt.axvline(
            best_step,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Best checkpoint (step {best_step})",
        )


# =============================================================================
# 3. 主流程
# =============================================================================

def main():

    if not STATE_FILE.is_file():
        raise FileNotFoundError(
            f"找不到：{STATE_FILE.resolve()}\n"
            "請先執行 python train_bert.py"
        )

    with STATE_FILE.open("r", encoding="utf-8") as file:
        state = json.load(file)

    logs = state.get("log_history", [])

    if not logs:
        raise RuntimeError("trainer_state.json 中沒有 log_history。")

    best_step = find_best_step(state)

    # -------------------------------------------------------------------------
    # Training / Validation Loss
    # -------------------------------------------------------------------------
    #
    # 判讀方式：
    #
    #     兩條線一起下降          → 模型持續學到有用的東西
    #     train 降、valid 回升    → 開始 overfitting（背答案）
    #
    # Early stopping 就是在 valid loss 回升一段時間後停下來。

    train_loss = extract_points(logs, "loss")
    valid_loss = extract_points(logs, "eval_loss")

    plt.figure(figsize=(10, 6))

    if train_loss:
        x, y = zip(*train_loss)
        plt.plot(x, y, label="Training Loss", linewidth=1.8)

    if valid_loss:
        x, y = zip(*valid_loss)
        plt.plot(x, y, marker="o", label="Validation Loss", linewidth=1.8)

    mark_best_step(best_step)

    plt.xlabel("Optimizer Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("BERT Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss 圖表：{LOSS_PLOT_FILE.resolve()}")

    # -------------------------------------------------------------------------
    # Validation Accuracy / Macro F1
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for key, name in METRIC_KEYS.items():
        points = extract_points(logs, key)
        if points:
            x, y = zip(*points)
            plt.plot(x, y, marker="o", label=name, linewidth=1.8)

    mark_best_step(best_step)

    plt.xlabel("Optimizer Step")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("BERT Validation Accuracy / Macro F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(METRIC_PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Metric 圖表：{METRIC_PLOT_FILE.resolve()}")

    print(f"Training loss points：{len(train_loss)}")
    print(f"Validation loss points：{len(valid_loss)}")
    print(f"Best step：{best_step}")


if __name__ == "__main__":
    main()
