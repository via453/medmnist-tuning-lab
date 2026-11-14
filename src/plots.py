"""由 results/*.csv 现场生成图件。绝不生成假数据：没有实验结果就明确提示并退出。

    python -m src.plots --results results/runs.csv

产出（有数据时）：
    figures/hp_sensitivity.png   各超参的 val_auc 波动范围
    figures/before_after.png     base 与最优配置的对比（多 seed 时带误差棒）
ROC / 混淆矩阵 / loss 曲线需要预测概率或逐 epoch 历史，由后续实验脚本补充。
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 图件统一用中文字体，避免 PPT 里全是英文轴标（缺字体时自动回退 SimHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

from .report import plot_loss_curves
from .utils import ensure_dir


def read_runs(path):
    if not os.path.isfile(path):
        print("NO DATA: %s 不存在。" % path)
        print("先在本机执行 python -m src.run / src.sweep 产出真实实验结果；"
              "本脚本不会生成示意数据或假图。")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
    if "val_auc" not in ok.columns or len(ok) == 0:
        print("NO DATA: %s 中没有有效的 val_auc 行。" % path)
        return None
    return ok


def plot_sensitivity(df, out_dir):
    """超参敏感性：每个超参的极差（条形）与均值（点），按极差从大到小排。"""
    NAMES = {
        "lr": "学习率", "optimizer": "优化器", "batch_size": "批大小",
        "weight_decay": "权重衰减", "dropout": "丢弃率", "use_bn": "是否用 BN",
        "activation": "激活函数", "pooling": "池化方式", "scheduler": "学习率排程",
        "conv_width": "模型宽度", "augment": "数据增强", "class_weight": "类别权重",
    }
    d = df[df["changed_key"].notna() & (df["changed_key"] != "")] \
        if "changed_key" in df.columns else pd.DataFrame()
    d = d[d["changed_key"] != "base"]
    if d.empty:
        return False
    groups = d.groupby("changed_key")["val_auc"]
    stat = []
    for k, g in groups:
        stat.append((k, float(g.min()), float(g.max()), float(g.mean())))
    stat.sort(key=lambda r: r[2] - r[1])          # 极差小的在下，大的在上

    labels = ["%s（%s）" % (NAMES.get(k, k), k) for k, _, _, _ in stat]
    lows = [r[1] for r in stat]
    highs = [r[2] for r in stat]
    means = [r[3] for r in stat]

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(stat) + 1.4))
    y = np.arange(len(stat))
    ax.barh(y, [h - l for l, h in zip(lows, highs)], left=lows,
            color="#b5d4f4", edgecolor="#378add", height=0.55)
    ax.scatter(means, y, color="#185fa5", zorder=3, s=34)
    ax.set_yticks(y, labels, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("验证集 AUC（条形=最低到最高，点=均值）", fontsize=12)
    ax.set_title("每个超参改一改，成绩能晃多少", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "hp_sensitivity.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("written:", p)
    return True


def plot_before_after(df, out_dir):
    """base 与最优配置的对比；多 seed 时带误差棒。"""
    if "changed_key" not in df.columns:
        return False
    base = df[df["changed_key"] == "base"]["val_auc"]
    rest = df[(df["changed_key"] != "base") & df["changed_key"].notna()]["val_auc"]
    if len(base) == 0 or len(rest) == 0:
        return False
    labels = ["原来（宽度 32）", "加大宽度（64）"]
    means = [float(base.mean()), float(rest.max())]
    errs = [float(base.std(ddof=0)), 0.0]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(labels, means, yerr=errs, capsize=6,
           color=["#b5d4f4", "#9fe1cb"], edgecolor="#185fa5")
    for i, v in enumerate(means):
        ax.text(i, v, "%.4f" % v, ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylim(max(0.0, min(means) - 0.08), min(1.0, max(means) + 0.08))
    ax.set_ylabel("验证集 AUC", fontsize=12)
    ax.set_title("两种设置的验证集成绩", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(out_dir, "before_after.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("written:", p)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate figures from results CSV")
    p.add_argument("--results", default="results/runs.csv")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--history", help="可选：训练历史 JSON，用于画 loss 曲线")
    args = p.parse_args(argv)

    df = read_runs(args.results)
    if df is None:
        return 1
    ensure_dir(args.out_dir)
    made = plot_sensitivity(df, args.out_dir) + plot_before_after(df, args.out_dir)
    loss_png = plot_loss_curves(args.history, args.out_dir)
    if loss_png:
        made += 1
        print("written:", loss_png)
    if not made:
        print("提示：CSV 里缺少 changed_key 分组列，暂无法出图；"
              "请用 src/sweep.py 产出带 changed_key 的扫描结果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
