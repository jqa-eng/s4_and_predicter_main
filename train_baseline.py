import os
import json
import torch
import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境时用无头后端
import matplotlib.pyplot as plt

from s4dd import S4forDenovoDesign
from s4dd.torch_callbacks import EarlyStopping, ModelCheckpoint, HistoryLogger

os.makedirs("./models_s4", exist_ok=True)

# 1) 构造模型（沿用你的设置；若显存吃紧可把 batch_size 调小）
s4 = S4forDenovoDesign(
    batch_size=1024,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 2) 训练（与原先一致）
history = s4.train(
    training_molecules_path="datasets/chemblv31/train.zip",
    val_molecules_path="datasets/chemblv31/valid.zip",
    callbacks=[
        EarlyStopping(patience=5, delta=1e-5, criterion="val_loss", mode="min"),
        ModelCheckpoint(save_fn=s4.save, save_per_epoch=3, basedir="./models/"),
        HistoryLogger(savedir="./models/")
    ]
)

# 3) 训练结束后，手动再保存一次最终模型（防止“最佳点”未被周期性checkpoint覆盖）
s4.save("./models/final_s4")

# 4) 取出训练曲线并绘图保存
# 优先使用 train(...) 的返回值；若返回值中没有，就回退读取 HistoryLogger 写入的文件
train_losses, val_losses = None, None

if isinstance(history, dict):
    train_losses = history.get("train_loss", None)
    val_losses = history.get("val_loss", None)

if train_losses is None or val_losses is None:
    # HistoryLogger 通常会在 ./models/ 下写出 history.json（或类似命名）
    # 你也可以打开 ./models/ 看看具体文件名，这里做一个常见命名的兜底尝试
    for fname in ["history.json", "train_history.json", "loss_history.json"]:
        path = os.path.join("./models", fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            train_losses = hist.get("train_loss", train_losses)
            val_losses = hist.get("val_loss", val_losses)
            break

# 防御：若历史仍为空，给出提示避免报错
if not train_losses or not val_losses:
    print("[WARN] 未能从返回值/日志文件中读取到训练历史，跳过绘图。")
else:
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="train_loss", linewidth=2)
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label="val_loss",   linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.title("S4 Training Curve (Train vs. Val)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out_png = "./models_s4/loss_curve.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] 训练曲线已保存：{out_png}")
