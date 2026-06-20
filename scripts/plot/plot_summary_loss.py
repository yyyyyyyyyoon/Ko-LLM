from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from project_paths import OUTPUT_ROOT

# Path
RUN_NAME = "summary_model"

LOG_FILE = OUTPUT_ROOT / RUN_NAME / "logs" / "train_log.csv"
PLOT_DIR = OUTPUT_ROOT / RUN_NAME / "plots"

if not LOG_FILE.exists():
    raise FileNotFoundError(f"train_log.csv not found: {LOG_FILE}")

PLOT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PNG = PLOT_DIR / "summary_model_train_loss.png"
EVAL_PNG = PLOT_DIR / "summary_model_eval_loss.png"
POINTS_CSV = PLOT_DIR / "summary_model_loss_points.csv"

# Load log
df = pd.read_csv(LOG_FILE)
print("Columns:", df.columns.tolist())

# Column detection
def find_col(candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Column not found. Candidates: {candidates}")


step_col = find_col(["optimizer_step", "global_step", "step"])
train_loss_col = find_col(["train_loss", "loss"])
eval_loss_col = find_col(["eval_loss", "validation_loss", "val_loss"])

# Numeric conversion
df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
df[train_loss_col] = pd.to_numeric(df[train_loss_col], errors="coerce")
df[eval_loss_col] = pd.to_numeric(df[eval_loss_col], errors="coerce")

train_df = df[[step_col, train_loss_col]].dropna()
eval_df = df[[step_col, eval_loss_col]].dropna()

save_df = pd.DataFrame({
    "step": df[step_col],
    "train_loss": df[train_loss_col],
    "eval_loss": df[eval_loss_col],
})
save_df.to_csv(POINTS_CSV, index=False, encoding="utf-8-sig")


# Plot 1. Summary Model Train Loss
train_df = train_df.sort_values(step_col).copy()

SMOOTH_WINDOW = 200
train_df["smooth_train_loss"] = train_df[train_loss_col].rolling(
    window=SMOOTH_WINDOW,
    min_periods=1
).mean()

max_step = int(train_df[step_col].max())

plt.figure(figsize=(10, 5))

plt.plot(
    train_df[step_col],
    train_df["smooth_train_loss"],
    label="Train Loss",
    linewidth=1.5,
)

plt.xlabel("Optimizer Step")
plt.ylabel("Loss")
plt.title(f"Summary Model Train Loss")
plt.ylim(1.8, 2.6)  # Summary loss가 잘 보이는 범위
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(TRAIN_PNG, dpi=300)
plt.show()

# Plot 2. Summary Model Eval Loss
plt.figure(figsize=(10, 5))
plt.plot(
    eval_df[step_col],
    eval_df[eval_loss_col],
    label="Eval Loss",
    marker="o",
    linewidth=1.4,
    markersize=3,
)

plt.xlabel("Optimizer Step")
plt.ylabel("Loss")
plt.title("Summary Model Eval Loss")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(EVAL_PNG, dpi=300)
plt.show()


print(f"Saved train loss figure: {TRAIN_PNG}")
print(f"Saved eval loss figure: {EVAL_PNG}")
print(f"Saved loss points csv: {POINTS_CSV}")